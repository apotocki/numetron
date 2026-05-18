// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <cstddef>
#include <cassert>
#include <memory>
#include <new>

namespace numetron::detail {

// stack_allocator_state<UpstreamAllocator>
//
// Thread-local stack-based memory resource.  Memory is served in LIFO order:
// each deallocate() must match the most recent not-yet-freed allocate().
//
// Internally it holds a singly-linked list of fixed-size chunks obtained from
// UpstreamAllocator.  Within each chunk a bump pointer advances on allocate()
// and retreats on deallocate().  When the current chunk is exhausted a new one
// is requested from the upstream allocator.  When a chunk becomes fully empty
// it is moved to an internal free-list instead of being returned to the upstream
// allocator, so it can be reused by the next allocate() call without touching
// the upstream at all.  Call clear() to release all free-list slabs back to
// the upstream allocator when you know they will no longer be needed.
//
// Template parameters:
//   UpstreamAllocator  -- must satisfy std::allocator requirements for `char`
//   ChunkBytes         -- number of T-sized elements per upstream chunk
//                         (chosen at compile time; tune for your workload)

template <typename UpstreamAllocator = std::allocator<char>,
          std::size_t ChunkBytes = 4096>
class stack_allocator_state
{
    using up_alloc  = typename std::allocator_traits<UpstreamAllocator>::
                        template rebind_alloc<char>;
    using up_traits = std::allocator_traits<up_alloc>;

    // Each slab obtained from the upstream allocator.
    // Layout: [slab header | data ... ]
    struct slab
    {
        slab*       prev;       // previous slab in the stack (nullptr = bottom)
        std::size_t capacity;   // usable bytes in this slab (excludes header)
        std::size_t used;       // bytes currently in use

        char* data() noexcept { return reinterpret_cast<char*>(this + 1); }
    };

    static constexpr std::size_t k_min_chunk = ChunkBytes;
    static constexpr std::size_t k_slab_overhead = sizeof(slab);

    up_alloc     m_upstream;
    slab*        m_top  = nullptr;  // current (topmost) active slab
    slab*        m_free = nullptr;  // linked list of empty (reusable) slabs

    // Allocate a new slab large enough for at least `bytes` usable bytes.
    slab* new_slab(std::size_t bytes)
    {
        std::size_t total = k_slab_overhead +
                            (std::max)(bytes, k_min_chunk - k_slab_overhead);
        char* raw = up_traits::allocate(m_upstream, total);
        slab* s   = ::new (raw) slab{};
        s->prev     = nullptr;
        s->capacity = total - k_slab_overhead;
        s->used     = 0;
        return s;
    }

    void free_slab(slab* s) noexcept
    {
        std::size_t total = k_slab_overhead + s->capacity;
        up_traits::deallocate(m_upstream, reinterpret_cast<char*>(s), total);
    }

public:
    explicit stack_allocator_state(UpstreamAllocator up = {})
        : m_upstream(std::move(up))
    {}

    stack_allocator_state(const stack_allocator_state&)            = delete;
    stack_allocator_state& operator=(const stack_allocator_state&) = delete;

    ~stack_allocator_state()
    {
        // All allocations must have been freed before destruction.
        assert(!m_top || m_top->used == 0);
        if (m_top) free_slab(m_top);
        clear();
    }

    // Release all slabs that are currently in the free-list back to the
    // upstream allocator.  Safe to call at any time; does not affect live
    // allocations.
    void clear() noexcept
    {
        while (m_free) {
            slab* s = m_free;
            m_free  = s->prev;
            free_slab(s);
        }
    }

    // Allocate `bytes` bytes aligned to `alignment`.
    [[nodiscard]] void* allocate(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
    {
        // Round up to alignment.
        if (!m_top || aligned_offset(m_top, alignment) + bytes > m_top->capacity) {
            // Try to reuse a slab from the free-list before going to upstream.
            slab* s = nullptr;
            slab** pp = &m_free;
            while (*pp) {
                if ((*pp)->capacity >= bytes + alignment) {
                    s   = *pp;
                    *pp = s->prev;  // unlink from free-list
                    break;
                }
                pp = &(*pp)->prev;
            }
            if (!s) {
                s = new_slab(k_slab_overhead + bytes + alignment);
            }
            s->prev = m_top;
            s->used = 0;
            m_top   = s;
        }

        std::size_t off = aligned_offset(m_top, alignment);
        m_top->used     = off + bytes;
        return m_top->data() + off;
    }

    // Deallocate must be called in strict LIFO order.
    void deallocate(void* ptr, std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) noexcept
    {
        assert(m_top);
        // The pointer must be at the top of the current slab.
        std::size_t off = aligned_offset_for(m_top, ptr);
        assert(m_top->data() + off == static_cast<char*>(ptr));
        assert(off + bytes == m_top->used);

        if (off == 0) {
            // Slab is now empty: move it to the free-list for later reuse.
            slab* s = m_top;
            m_top   = s->prev;
            s->used = 0;
            s->prev = m_free;
            m_free  = s;
        } else {
            m_top->used = off;
        }
    }

private:
    // Offset within slab->data() at which an aligned allocation would start.
    static std::size_t aligned_offset(slab* s, std::size_t alignment) noexcept
    {
        std::uintptr_t base = reinterpret_cast<std::uintptr_t>(s->data()) + s->used;
        std::uintptr_t aligned = (base + alignment - 1) & ~(alignment - 1);
        return s->used + static_cast<std::size_t>(aligned - base);
    }

    // Recover the offset of a pointer that was previously allocated from slab s.
    static std::size_t aligned_offset_for(slab* s, void* ptr) noexcept
    {
        return static_cast<std::size_t>(
            static_cast<char*>(ptr) - s->data());
    }
};

// stack_allocator<T>
//
// std::allocator-compatible adaptor that delegates to a stack_allocator_state.
// Multiple stack_allocator instances referring to the same state can coexist;
// the state itself owns the memory.
//
// Typical use with a thread_local state:
//
//   using ScratchAlloc = numetron::detail::stack_allocator<uint64_t>;
//   ScratchAlloc alloc;   // refers to the thread-local state automatically
//
template <typename T,
          typename UpstreamAllocator = std::allocator<char>,
          std::size_t ChunkBytes = 4096>
class stack_allocator
{
public:
    using value_type      = T;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using state_type = stack_allocator_state<UpstreamAllocator, ChunkBytes>;

    // Rebind: keep the same state_type type, just change T.
    template <typename U>
    struct rebind { using other = stack_allocator<U, UpstreamAllocator, ChunkBytes>; };

    // Construct from an explicit state reference.
    explicit stack_allocator(state_type& state) noexcept : state_(&state) {}

    // Default constructor: uses a thread-local state instance.
    stack_allocator() noexcept : state_(&thread_local_state()) {}

    // Rebinding constructor (required by allocator_traits).
    template <typename U>
    stack_allocator(const stack_allocator<U, UpstreamAllocator, ChunkBytes>& other) noexcept
        : state_(other.state_)
    {}

    T* allocate(std::size_t n)
    {
        return static_cast<T*>(state_->allocate(n * sizeof(T), alignof(T)));
    }

    void deallocate(T* p, std::size_t n) noexcept
    {
        state_->deallocate(p, n * sizeof(T), alignof(T));
    }

    // Two allocators are equal iff they share the same state.
    friend bool operator==(const stack_allocator& a, const stack_allocator& b) noexcept
    {
        return a.state_ == b.state_;
    }

    // Access the underlying state (needed by the rebinding constructor).
    state_type* state_;

private:
    static state_type& thread_local_state()
    {
        thread_local state_type tl_state;
        return tl_state;
    }
};

} // namespace numetron::detail
