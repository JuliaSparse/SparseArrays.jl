# This file is a part of Julia. License is MIT: https://julialang.org/license

import Serialization

# Internal locking for the solver factorizations (UMFPACK, CHOLMOD, SPQR). Not API.
#
# Each factorization type `T` stores its lock in the field `_lock` and defines
# `_factorlock(F::T) = getfield(F, :_lock)`. A call that reads the factorization's mutable
# state runs under `@_readlock F expr`, and a call that writes it under
# `@_writelock F expr`. The lock is either a `FactorLock`, a readers-writer lock that lets
# reads of one factorization run in parallel, or a plain `ReentrantLock`, for which both
# macros take the lock exclusively; call sites say which mode they need either way.
#
# Rules:
# - Never hold the locks of two factorizations at once. Copy-then-mutate code reads the
#   source, releases it, and then locks the new object.
# - Finalizers never lock.
# - Public methods are thin `@_readlock F _kernel(F)` wrappers around unlocked kernels
#   that assume the caller holds the lock, so that internal paths take it once.
# - A task holding a read lock cannot upgrade it: `@_writelock` throws
#   `ConcurrencyViolationError` instead of deadlocking. Take the write lock up front.

# `FactorLock()` is a writer-preferring readers-writer lock, reentrant per task:
# - A task holding the write lock may take the write lock or a read lock again.
# - A task holding a read lock may take another read lock, even while a writer waits.
# - Any other reader waits while a writer holds the lock or is waiting for it, so a
#   stream of readers cannot starve a writer.
# - Taking the write lock while holding only a read lock throws
#   `ConcurrencyViolationError`, as does releasing a lock the task does not hold.
# - A serialized `FactorLock` deserializes to a fresh, unlocked lock.
mutable struct FactorLock
    const cond::Threads.Condition
    writer::Union{Nothing,Task}
    writer_depth::Int
    # A multiset: a task appears once for each read lock it holds.
    const readers::Vector{Task}
    waiting_writers::Int
end

# The size hint keeps the uncontended path from allocating.
FactorLock() = FactorLock(Threads.Condition(), nothing, 0, sizehint!(Task[], 8), 0)

_factorlock(l::Union{FactorLock,ReentrantLock}) = l

function _findread(l::FactorLock, t::Task)
    for i in lastindex(l.readers):-1:firstindex(l.readers)
        l.readers[i] === t && return i
    end
    return 0
end

function _rdlock(l::FactorLock)
    t = current_task()
    lock(l.cond)
    try
        if l.writer !== t && _findread(l, t) == 0
            while l.writer !== nothing || l.waiting_writers > 0
                wait(l.cond)
            end
        end
        push!(l.readers, t)
    finally
        unlock(l.cond)
    end
    return nothing
end

function _rdunlock(l::FactorLock)
    t = current_task()
    lock(l.cond)
    try
        i = _findread(l, t)
        i == 0 &&
            throw(ConcurrencyViolationError("FactorLock: read unlock by a task that holds no read lock"))
        deleteat!(l.readers, i)
        isempty(l.readers) && notify(l.cond)
    finally
        unlock(l.cond)
    end
    return nothing
end

function _wrlock(l::FactorLock)
    t = current_task()
    lock(l.cond)
    try
        if l.writer === t
            l.writer_depth += 1
            return nothing
        end
        _findread(l, t) != 0 &&
            throw(ConcurrencyViolationError("FactorLock: cannot upgrade a read lock to a write lock; take the write lock first"))
        if l.writer !== nothing || !isempty(l.readers)
            l.waiting_writers += 1
            try
                while l.writer !== nothing || !isempty(l.readers)
                    wait(l.cond)
                end
                l.writer = t
            finally
                l.waiting_writers -= 1
                # A writer interrupted while waiting may have been the only thing holding
                # readers back, so wake them.
                l.writer === t || notify(l.cond)
            end
        else
            l.writer = t
        end
        l.writer_depth = 1
    finally
        unlock(l.cond)
    end
    return nothing
end

function _wrunlock(l::FactorLock)
    t = current_task()
    lock(l.cond)
    try
        l.writer === t ||
            throw(ConcurrencyViolationError("FactorLock: write unlock by a task that does not hold the write lock"))
        l.writer_depth -= 1
        if l.writer_depth == 0
            l.writer = nothing
            notify(l.cond)
        end
    finally
        unlock(l.cond)
    end
    return nothing
end

# A plain `ReentrantLock` has only an exclusive mode.
_rdlock(l::ReentrantLock) = lock(l)
_rdunlock(l::ReentrantLock) = unlock(l)
_wrlock(l::ReentrantLock) = lock(l)
_wrunlock(l::ReentrantLock) = unlock(l)

# Macros rather than do-blocks: a closure would box the locals that `expr` reassigns.
# Like `@lock`, they acquire before the `try`, so a failed acquisition releases nothing.
# `@_readlock F expr` evaluates `expr` holding the read lock of the factorization `F`
# (or of the lock `F`); `@_writelock F expr` likewise with the write lock.
macro _readlock(F, expr)
    quote
        l = _factorlock($(esc(F)))
        _rdlock(l)
        try
            $(esc(expr))
        finally
            _rdunlock(l)
        end
    end
end

macro _writelock(F, expr)
    quote
        l = _factorlock($(esc(F)))
        _wrlock(l)
        try
            $(esc(expr))
        finally
            _wrunlock(l)
        end
    end
end

# The lock state of a factorization must not travel with it.
Serialization.serialize(s::Serialization.AbstractSerializer, ::FactorLock) =
    Serialization.serialize_type(s, FactorLock)
Serialization.deserialize(::Serialization.AbstractSerializer, ::Type{FactorLock}) = FactorLock()

# For tests: the number of read locks held, whether a writer holds the lock, and the
# number of writers waiting.
_nreaders(l::FactorLock) = @lock l.cond length(l.readers)
_haswriter(l::FactorLock) = @lock l.cond l.writer !== nothing
_nwaiting(l::FactorLock) = @lock l.cond l.waiting_writers
