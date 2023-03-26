function SuiteSparse_config_printf_func_get()
    ccall((:SuiteSparse_config_printf_func_get, libcholmod), Ptr{Cvoid}, ())
end

function SuiteSparse_config_malloc_func_get()
    ccall((:SuiteSparse_config_malloc_func_get, libcholmod), Ptr{Cvoid}, ())
end

function SuiteSparse_config_calloc_func_get()
    ccall((:SuiteSparse_config_calloc_func_get, libcholmod), Ptr{Cvoid}, ())
end

function SuiteSparse_config_realloc_func_get()
    ccall((:SuiteSparse_config_realloc_func_get, libcholmod), Ptr{Cvoid}, ())
end

function SuiteSparse_config_free_func_get()
    ccall((:SuiteSparse_config_free_func_get, libcholmod), Ptr{Cvoid}, ())
end

function SuiteSparse_config_hypot_func_get()
    ccall((:SuiteSparse_config_hypot_func_get, libcholmod), Ptr{Cvoid}, ())
end

function SuiteSparse_config_divcomplex_func_get()
    ccall((:SuiteSparse_config_divcomplex_func_get, libcholmod), Ptr{Cvoid}, ())
end

function SuiteSparse_config_malloc_func_set(malloc_func)
    ccall((:SuiteSparse_config_malloc_func_set, libcholmod), Cvoid, (Ptr{Cvoid},), malloc_func)
end

function SuiteSparse_config_calloc_func_set(calloc_func)
    ccall((:SuiteSparse_config_calloc_func_set, libcholmod), Cvoid, (Ptr{Cvoid},), calloc_func)
end

function SuiteSparse_config_realloc_func_set(realloc_func)
    ccall((:SuiteSparse_config_realloc_func_set, libcholmod), Cvoid, (Ptr{Cvoid},), realloc_func)
end

function SuiteSparse_config_free_func_set(free_func)
    ccall((:SuiteSparse_config_free_func_set, libcholmod), Cvoid, (Ptr{Cvoid},), free_func)
end

function SuiteSparse_config_printf_func_set(printf_func)
    ccall((:SuiteSparse_config_printf_func_set, libcholmod), Cvoid, (Ptr{Cvoid},), printf_func)
end

function SuiteSparse_config_hypot_func_set(hypot_func)
    ccall((:SuiteSparse_config_hypot_func_set, libcholmod), Cvoid, (Ptr{Cvoid},), hypot_func)
end

function SuiteSparse_config_divcomplex_func_set(divcomplex_func)
    ccall((:SuiteSparse_config_divcomplex_func_set, libcholmod), Cvoid, (Ptr{Cvoid},), divcomplex_func)
end

function SuiteSparse_config_malloc(s)
    ccall((:SuiteSparse_config_malloc, libcholmod), Ptr{Cvoid}, (Csize_t,), s)
end

function SuiteSparse_config_calloc(n, s)
    ccall((:SuiteSparse_config_calloc, libcholmod), Ptr{Cvoid}, (Csize_t, Csize_t), n, s)
end

function SuiteSparse_config_realloc(arg1, s)
    ccall((:SuiteSparse_config_realloc, libcholmod), Ptr{Cvoid}, (Ptr{Cvoid}, Csize_t), arg1, s)
end

function SuiteSparse_config_free(arg1)
    ccall((:SuiteSparse_config_free, libcholmod), Cvoid, (Ptr{Cvoid},), arg1)
end

function SuiteSparse_config_hypot(x, y)
    ccall((:SuiteSparse_config_hypot, libcholmod), Cdouble, (Cdouble, Cdouble), x, y)
end

function SuiteSparse_config_divcomplex(xr, xi, yr, yi, zr, zi)
    ccall((:SuiteSparse_config_divcomplex, libcholmod), Cint, (Cdouble, Cdouble, Cdouble, Cdouble, Ptr{Cdouble}, Ptr{Cdouble}), xr, xi, yr, yi, zr, zi)
end

function SuiteSparse_start()
    ccall((:SuiteSparse_start, libcholmod), Cvoid, ())
end

function SuiteSparse_finish()
    ccall((:SuiteSparse_finish, libcholmod), Cvoid, ())
end

function SuiteSparse_malloc(nitems, size_of_item)
    ccall((:SuiteSparse_malloc, libcholmod), Ptr{Cvoid}, (Csize_t, Csize_t), nitems, size_of_item)
end

function SuiteSparse_calloc(nitems, size_of_item)
    ccall((:SuiteSparse_calloc, libcholmod), Ptr{Cvoid}, (Csize_t, Csize_t), nitems, size_of_item)
end

function SuiteSparse_realloc(nitems_new, nitems_old, size_of_item, p, ok)
    ccall((:SuiteSparse_realloc, libcholmod), Ptr{Cvoid}, (Csize_t, Csize_t, Csize_t, Ptr{Cvoid}, Ptr{Cint}), nitems_new, nitems_old, size_of_item, p, ok)
end

function SuiteSparse_free(p)
    ccall((:SuiteSparse_free, libcholmod), Ptr{Cvoid}, (Ptr{Cvoid},), p)
end

function SuiteSparse_tic(tic)
    ccall((:SuiteSparse_tic, libcholmod), Cvoid, (Ptr{Cdouble},), tic)
end

function SuiteSparse_toc(tic)
    ccall((:SuiteSparse_toc, libcholmod), Cdouble, (Ptr{Cdouble},), tic)
end

function SuiteSparse_time()
    ccall((:SuiteSparse_time, libcholmod), Cdouble, ())
end

function SuiteSparse_hypot(x, y)
    ccall((:SuiteSparse_hypot, libcholmod), Cdouble, (Cdouble, Cdouble), x, y)
end

function SuiteSparse_divcomplex(ar, ai, br, bi, cr, ci)
    ccall((:SuiteSparse_divcomplex, libcholmod), Cint, (Cdouble, Cdouble, Cdouble, Cdouble, Ptr{Cdouble}, Ptr{Cdouble}), ar, ai, br, bi, cr, ci)
end

function SuiteSparse_version(version)
    ccall((:SuiteSparse_version, libcholmod), Cint, (Ptr{Cint},), version)
end

function SuiteSparse_BLAS_library()
    ccall((:SuiteSparse_BLAS_library, libcholmod), Ptr{Cchar}, ())
end

function SuiteSparse_BLAS_integer_size()
    ccall((:SuiteSparse_BLAS_integer_size, libcholmod), Csize_t, ())
end

struct cholmod_method_struct
    lnz::Cdouble
    fl::Cdouble
    prune_dense::Cdouble
    prune_dense2::Cdouble
    nd_oksep::Cdouble
    other_1::NTuple{4, Cdouble}
    nd_small::Csize_t
    other_2::NTuple{4, Csize_t}
    aggressive::Cint
    order_for_lu::Cint
    nd_compress::Cint
    nd_camd::Cint
    nd_components::Cint
    ordering::Cint
    other_3::NTuple{4, Csize_t}
end

mutable struct cholmod_common_struct
    dbound::Cdouble
    grow0::Cdouble
    grow1::Cdouble
    grow2::Csize_t
    maxrank::Csize_t
    supernodal_switch::Cdouble
    supernodal::Cint
    final_asis::Cint
    final_super::Cint
    final_ll::Cint
    final_pack::Cint
    final_monotonic::Cint
    final_resymbol::Cint
    zrelax::NTuple{3, Cdouble}
    nrelax::NTuple{3, Csize_t}
    prefer_zomplex::Cint
    prefer_upper::Cint
    quick_return_if_not_posdef::Cint
    prefer_binary::Cint
    print::Cint
    precise::Cint
    try_catch::Cint
    error_handler::Ptr{Cvoid}
    nmethods::Cint
    current::Cint
    selected::Cint
    method::NTuple{10, cholmod_method_struct}
    postorder::Cint
    default_nesdis::Cint
    metis_memory::Cdouble
    metis_dswitch::Cdouble
    metis_nswitch::Csize_t
    nrow::Csize_t
    mark::Int64
    iworksize::Csize_t
    xworksize::Csize_t
    Flag::Ptr{Cvoid}
    Head::Ptr{Cvoid}
    Xwork::Ptr{Cvoid}
    Iwork::Ptr{Cvoid}
    itype::Cint
    dtype::Cint
    no_workspace_reallocate::Cint
    status::Cint
    fl::Cdouble
    lnz::Cdouble
    anz::Cdouble
    modfl::Cdouble
    malloc_count::Csize_t
    memory_usage::Csize_t
    memory_inuse::Csize_t
    nrealloc_col::Cdouble
    nrealloc_factor::Cdouble
    ndbounds_hit::Cdouble
    rowfacfl::Cdouble
    aatfl::Cdouble
    called_nd::Cint
    blas_ok::Cint
    SPQR_grain::Cdouble
    SPQR_small::Cdouble
    SPQR_shrink::Cint
    SPQR_nthreads::Cint
    SPQR_flopcount::Cdouble
    SPQR_analyze_time::Cdouble
    SPQR_factorize_time::Cdouble
    SPQR_solve_time::Cdouble
    SPQR_flopcount_bound::Cdouble
    SPQR_tol_used::Cdouble
    SPQR_norm_E_fro::Cdouble
    SPQR_istat::NTuple{10, Int64}
    useGPU::Cint
    maxGpuMemBytes::Csize_t
    maxGpuMemFraction::Cdouble
    gpuMemorySize::Csize_t
    gpuKernelTime::Cdouble
    gpuFlops::Int64
    gpuNumKernelLaunches::Cint
    cublasHandle::Ptr{Cvoid}
    gpuStream::NTuple{8, Ptr{Cvoid}}
    cublasEventPotrf::NTuple{3, Ptr{Cvoid}}
    updateCKernelsComplete::Ptr{Cvoid}
    updateCBuffersFree::NTuple{8, Ptr{Cvoid}}
    dev_mempool::Ptr{Cvoid}
    dev_mempool_size::Csize_t
    host_pinned_mempool::Ptr{Cvoid}
    host_pinned_mempool_size::Csize_t
    devBuffSize::Csize_t
    ibuffer::Cint
    syrkStart::Cdouble
    cholmod_cpu_gemm_time::Cdouble
    cholmod_cpu_syrk_time::Cdouble
    cholmod_cpu_trsm_time::Cdouble
    cholmod_cpu_potrf_time::Cdouble
    cholmod_gpu_gemm_time::Cdouble
    cholmod_gpu_syrk_time::Cdouble
    cholmod_gpu_trsm_time::Cdouble
    cholmod_gpu_potrf_time::Cdouble
    cholmod_assemble_time::Cdouble
    cholmod_assemble_time2::Cdouble
    cholmod_cpu_gemm_calls::Csize_t
    cholmod_cpu_syrk_calls::Csize_t
    cholmod_cpu_trsm_calls::Csize_t
    cholmod_cpu_potrf_calls::Csize_t
    cholmod_gpu_gemm_calls::Csize_t
    cholmod_gpu_syrk_calls::Csize_t
    cholmod_gpu_trsm_calls::Csize_t
    cholmod_gpu_potrf_calls::Csize_t
    chunk::Cdouble
    nthreads_max::Cint
    cholmod_common_struct() = new()
end

const cholmod_common = cholmod_common_struct

function cholmod_start(Common)
    ccall((:cholmod_start, libcholmod), Cint, (Ptr{cholmod_common},), Common)
end

function cholmod_l_start(arg1)
    ccall((:cholmod_l_start, libcholmod), Cint, (Ptr{cholmod_common},), arg1)
end

function cholmod_finish(Common)
    ccall((:cholmod_finish, libcholmod), Cint, (Ptr{cholmod_common},), Common)
end

function cholmod_l_finish(arg1)
    ccall((:cholmod_l_finish, libcholmod), Cint, (Ptr{cholmod_common},), arg1)
end

function cholmod_defaults(Common)
    ccall((:cholmod_defaults, libcholmod), Cint, (Ptr{cholmod_common},), Common)
end

function cholmod_l_defaults(arg1)
    ccall((:cholmod_l_defaults, libcholmod), Cint, (Ptr{cholmod_common},), arg1)
end

function cholmod_maxrank(n, Common)
    ccall((:cholmod_maxrank, libcholmod), Csize_t, (Csize_t, Ptr{cholmod_common}), n, Common)
end

function cholmod_l_maxrank(arg1, arg2)
    ccall((:cholmod_l_maxrank, libcholmod), Csize_t, (Csize_t, Ptr{cholmod_common}), arg1, arg2)
end

function cholmod_allocate_work(nrow, iworksize, xworksize, Common)
    ccall((:cholmod_allocate_work, libcholmod), Cint, (Csize_t, Csize_t, Csize_t, Ptr{cholmod_common}), nrow, iworksize, xworksize, Common)
end

function cholmod_l_allocate_work(arg1, arg2, arg3, arg4)
    ccall((:cholmod_l_allocate_work, libcholmod), Cint, (Csize_t, Csize_t, Csize_t, Ptr{cholmod_common}), arg1, arg2, arg3, arg4)
end

function cholmod_free_work(Common)
    ccall((:cholmod_free_work, libcholmod), Cint, (Ptr{cholmod_common},), Common)
end

function cholmod_l_free_work(arg1)
    ccall((:cholmod_l_free_work, libcholmod), Cint, (Ptr{cholmod_common},), arg1)
end

function cholmod_clear_flag(Common)
    ccall((:cholmod_clear_flag, libcholmod), Int64, (Ptr{cholmod_common},), Common)
end

function cholmod_l_clear_flag(arg1)
    ccall((:cholmod_l_clear_flag, libcholmod), Int64, (Ptr{cholmod_common},), arg1)
end

function cholmod_error(status, file, line, message, Common)
    ccall((:cholmod_error, libcholmod), Cint, (Cint, Ptr{Cchar}, Cint, Ptr{Cchar}, Ptr{cholmod_common}), status, file, line, message, Common)
end

function cholmod_l_error(arg1, arg2, arg3, arg4, arg5)
    ccall((:cholmod_l_error, libcholmod), Cint, (Cint, Ptr{Cchar}, Cint, Ptr{Cchar}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5)
end

function cholmod_dbound(dj, Common)
    ccall((:cholmod_dbound, libcholmod), Cdouble, (Cdouble, Ptr{cholmod_common}), dj, Common)
end

function cholmod_l_dbound(arg1, arg2)
    ccall((:cholmod_l_dbound, libcholmod), Cdouble, (Cdouble, Ptr{cholmod_common}), arg1, arg2)
end

function cholmod_hypot(x, y)
    ccall((:cholmod_hypot, libcholmod), Cdouble, (Cdouble, Cdouble), x, y)
end

function cholmod_l_hypot(arg1, arg2)
    ccall((:cholmod_l_hypot, libcholmod), Cdouble, (Cdouble, Cdouble), arg1, arg2)
end

function cholmod_divcomplex(ar, ai, br, bi, cr, ci)
    ccall((:cholmod_divcomplex, libcholmod), Cint, (Cdouble, Cdouble, Cdouble, Cdouble, Ptr{Cdouble}, Ptr{Cdouble}), ar, ai, br, bi, cr, ci)
end

function cholmod_l_divcomplex(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:cholmod_l_divcomplex, libcholmod), Cint, (Cdouble, Cdouble, Cdouble, Cdouble, Ptr{Cdouble}, Ptr{Cdouble}), arg1, arg2, arg3, arg4, arg5, arg6)
end

mutable struct cholmod_sparse_struct
    nrow::Csize_t
    ncol::Csize_t
    nzmax::Csize_t
    p::Ptr{Cvoid}
    i::Ptr{Cvoid}
    nz::Ptr{Cvoid}
    x::Ptr{Cvoid}
    z::Ptr{Cvoid}
    stype::Cint
    itype::Cint
    xtype::Cint
    dtype::Cint
    sorted::Cint
    packed::Cint
    cholmod_sparse_struct() = new()
end

const cholmod_sparse = cholmod_sparse_struct

mutable struct cholmod_descendant_score_t
    score::Cdouble
    d::Int64
    cholmod_descendant_score_t() = new()
end

const descendantScore = cholmod_descendant_score_t

function cholmod_score_comp(i, j)
    ccall((:cholmod_score_comp, libcholmod), Cint, (Ptr{cholmod_descendant_score_t}, Ptr{cholmod_descendant_score_t}), i, j)
end

function cholmod_l_score_comp(i, j)
    ccall((:cholmod_l_score_comp, libcholmod), Cint, (Ptr{cholmod_descendant_score_t}, Ptr{cholmod_descendant_score_t}), i, j)
end

function cholmod_allocate_sparse(nrow, ncol, nzmax, sorted, packed, stype, xtype, Common)
    ccall((:cholmod_allocate_sparse, libcholmod), Ptr{cholmod_sparse}, (Csize_t, Csize_t, Csize_t, Cint, Cint, Cint, Cint, Ptr{cholmod_common}), nrow, ncol, nzmax, sorted, packed, stype, xtype, Common)
end

function cholmod_l_allocate_sparse(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
    ccall((:cholmod_l_allocate_sparse, libcholmod), Ptr{cholmod_sparse}, (Csize_t, Csize_t, Csize_t, Cint, Cint, Cint, Cint, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
end

function cholmod_free_sparse(A, Common)
    ccall((:cholmod_free_sparse, libcholmod), Cint, (Ptr{Ptr{cholmod_sparse}}, Ptr{cholmod_common}), A, Common)
end

function cholmod_l_free_sparse(arg1, arg2)
    ccall((:cholmod_l_free_sparse, libcholmod), Cint, (Ptr{Ptr{cholmod_sparse}}, Ptr{cholmod_common}), arg1, arg2)
end

function cholmod_reallocate_sparse(nznew, A, Common)
    ccall((:cholmod_reallocate_sparse, libcholmod), Cint, (Csize_t, Ptr{cholmod_sparse}, Ptr{cholmod_common}), nznew, A, Common)
end

function cholmod_l_reallocate_sparse(arg1, arg2, arg3)
    ccall((:cholmod_l_reallocate_sparse, libcholmod), Cint, (Csize_t, Ptr{cholmod_sparse}, Ptr{cholmod_common}), arg1, arg2, arg3)
end

function cholmod_nnz(A, Common)
    ccall((:cholmod_nnz, libcholmod), Int64, (Ptr{cholmod_sparse}, Ptr{cholmod_common}), A, Common)
end

function cholmod_l_nnz(arg1, arg2)
    ccall((:cholmod_l_nnz, libcholmod), Int64, (Ptr{cholmod_sparse}, Ptr{cholmod_common}), arg1, arg2)
end

function cholmod_speye(nrow, ncol, xtype, Common)
    ccall((:cholmod_speye, libcholmod), Ptr{cholmod_sparse}, (Csize_t, Csize_t, Cint, Ptr{cholmod_common}), nrow, ncol, xtype, Common)
end

function cholmod_l_speye(arg1, arg2, arg3, arg4)
    ccall((:cholmod_l_speye, libcholmod), Ptr{cholmod_sparse}, (Csize_t, Csize_t, Cint, Ptr{cholmod_common}), arg1, arg2, arg3, arg4)
end

function cholmod_spzeros(nrow, ncol, nzmax, xtype, Common)
    ccall((:cholmod_spzeros, libcholmod), Ptr{cholmod_sparse}, (Csize_t, Csize_t, Csize_t, Cint, Ptr{cholmod_common}), nrow, ncol, nzmax, xtype, Common)
end

function cholmod_l_spzeros(arg1, arg2, arg3, arg4, arg5)
    ccall((:cholmod_l_spzeros, libcholmod), Ptr{cholmod_sparse}, (Csize_t, Csize_t, Csize_t, Cint, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5)
end

function cholmod_transpose(A, values, Common)
    ccall((:cholmod_transpose, libcholmod), Ptr{cholmod_sparse}, (Ptr{cholmod_sparse}, Cint, Ptr{cholmod_common}), A, values, Common)
end

function cholmod_l_transpose(arg1, arg2, arg3)
    ccall((:cholmod_l_transpose, libcholmod), Ptr{cholmod_sparse}, (Ptr{cholmod_sparse}, Cint, Ptr{cholmod_common}), arg1, arg2, arg3)
end

function cholmod_transpose_unsym(A, values, Perm, fset, fsize, F, Common)
    ccall((:cholmod_transpose_unsym, libcholmod), Cint, (Ptr{cholmod_sparse}, Cint, Ptr{Int32}, Ptr{Int32}, Csize_t, Ptr{cholmod_sparse}, Ptr{cholmod_common}), A, values, Perm, fset, fsize, F, Common)
end

function cholmod_l_transpose_unsym(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    ccall((:cholmod_l_transpose_unsym, libcholmod), Cint, (Ptr{cholmod_sparse}, Cint, Ptr{Int64}, Ptr{Int64}, Csize_t, Ptr{cholmod_sparse}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6, arg7)
end

function cholmod_transpose_sym(A, values, Perm, F, Common)
    ccall((:cholmod_transpose_sym, libcholmod), Cint, (Ptr{cholmod_sparse}, Cint, Ptr{Int32}, Ptr{cholmod_sparse}, Ptr{cholmod_common}), A, values, Perm, F, Common)
end

function cholmod_l_transpose_sym(arg1, arg2, arg3, arg4, arg5)
    ccall((:cholmod_l_transpose_sym, libcholmod), Cint, (Ptr{cholmod_sparse}, Cint, Ptr{Int64}, Ptr{cholmod_sparse}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5)
end

function cholmod_ptranspose(A, values, Perm, fset, fsize, Common)
    ccall((:cholmod_ptranspose, libcholmod), Ptr{cholmod_sparse}, (Ptr{cholmod_sparse}, Cint, Ptr{Int32}, Ptr{Int32}, Csize_t, Ptr{cholmod_common}), A, values, Perm, fset, fsize, Common)
end

function cholmod_l_ptranspose(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:cholmod_l_ptranspose, libcholmod), Ptr{cholmod_sparse}, (Ptr{cholmod_sparse}, Cint, Ptr{Int64}, Ptr{Int64}, Csize_t, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6)
end

function cholmod_sort(A, Common)
    ccall((:cholmod_sort, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{cholmod_common}), A, Common)
end

function cholmod_l_sort(arg1, arg2)
    ccall((:cholmod_l_sort, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{cholmod_common}), arg1, arg2)
end

function cholmod_band(A, k1, k2, mode, Common)
    ccall((:cholmod_band, libcholmod), Ptr{cholmod_sparse}, (Ptr{cholmod_sparse}, Int64, Int64, Cint, Ptr{cholmod_common}), A, k1, k2, mode, Common)
end

function cholmod_l_band(arg1, arg2, arg3, arg4, arg5)
    ccall((:cholmod_l_band, libcholmod), Ptr{cholmod_sparse}, (Ptr{cholmod_sparse}, Int64, Int64, Cint, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5)
end

function cholmod_band_inplace(k1, k2, mode, A, Common)
    ccall((:cholmod_band_inplace, libcholmod), Cint, (Int64, Int64, Cint, Ptr{cholmod_sparse}, Ptr{cholmod_common}), k1, k2, mode, A, Common)
end

function cholmod_l_band_inplace(arg1, arg2, arg3, arg4, arg5)
    ccall((:cholmod_l_band_inplace, libcholmod), Cint, (Int64, Int64, Cint, Ptr{cholmod_sparse}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5)
end

function cholmod_aat(A, fset, fsize, mode, Common)
    ccall((:cholmod_aat, libcholmod), Ptr{cholmod_sparse}, (Ptr{cholmod_sparse}, Ptr{Int32}, Csize_t, Cint, Ptr{cholmod_common}), A, fset, fsize, mode, Common)
end

function cholmod_l_aat(arg1, arg2, arg3, arg4, arg5)
    ccall((:cholmod_l_aat, libcholmod), Ptr{cholmod_sparse}, (Ptr{cholmod_sparse}, Ptr{Int64}, Csize_t, Cint, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5)
end

function cholmod_copy_sparse(A, Common)
    ccall((:cholmod_copy_sparse, libcholmod), Ptr{cholmod_sparse}, (Ptr{cholmod_sparse}, Ptr{cholmod_common}), A, Common)
end

function cholmod_l_copy_sparse(arg1, arg2)
    ccall((:cholmod_l_copy_sparse, libcholmod), Ptr{cholmod_sparse}, (Ptr{cholmod_sparse}, Ptr{cholmod_common}), arg1, arg2)
end

function cholmod_copy(A, stype, mode, Common)
    ccall((:cholmod_copy, libcholmod), Ptr{cholmod_sparse}, (Ptr{cholmod_sparse}, Cint, Cint, Ptr{cholmod_common}), A, stype, mode, Common)
end

function cholmod_l_copy(arg1, arg2, arg3, arg4)
    ccall((:cholmod_l_copy, libcholmod), Ptr{cholmod_sparse}, (Ptr{cholmod_sparse}, Cint, Cint, Ptr{cholmod_common}), arg1, arg2, arg3, arg4)
end

function cholmod_add(A, B, alpha, beta, values, sorted, Common)
    ccall((:cholmod_add, libcholmod), Ptr{cholmod_sparse}, (Ptr{cholmod_sparse}, Ptr{cholmod_sparse}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint, Ptr{cholmod_common}), A, B, alpha, beta, values, sorted, Common)
end

function cholmod_l_add(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    ccall((:cholmod_l_add, libcholmod), Ptr{cholmod_sparse}, (Ptr{cholmod_sparse}, Ptr{cholmod_sparse}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6, arg7)
end

function cholmod_sparse_xtype(to_xtype, A, Common)
    ccall((:cholmod_sparse_xtype, libcholmod), Cint, (Cint, Ptr{cholmod_sparse}, Ptr{cholmod_common}), to_xtype, A, Common)
end

function cholmod_l_sparse_xtype(arg1, arg2, arg3)
    ccall((:cholmod_l_sparse_xtype, libcholmod), Cint, (Cint, Ptr{cholmod_sparse}, Ptr{cholmod_common}), arg1, arg2, arg3)
end

mutable struct cholmod_factor_struct
    n::Csize_t
    minor::Csize_t
    Perm::Ptr{Cvoid}
    ColCount::Ptr{Cvoid}
    IPerm::Ptr{Cvoid}
    nzmax::Csize_t
    p::Ptr{Cvoid}
    i::Ptr{Cvoid}
    x::Ptr{Cvoid}
    z::Ptr{Cvoid}
    nz::Ptr{Cvoid}
    next::Ptr{Cvoid}
    prev::Ptr{Cvoid}
    nsuper::Csize_t
    ssize::Csize_t
    xsize::Csize_t
    maxcsize::Csize_t
    maxesize::Csize_t
    super::Ptr{Cvoid}
    pi::Ptr{Cvoid}
    px::Ptr{Cvoid}
    s::Ptr{Cvoid}
    ordering::Cint
    is_ll::Cint
    is_super::Cint
    is_monotonic::Cint
    itype::Cint
    xtype::Cint
    dtype::Cint
    useGPU::Cint
    cholmod_factor_struct() = new()
end

const cholmod_factor = cholmod_factor_struct

function cholmod_allocate_factor(n, Common)
    ccall((:cholmod_allocate_factor, libcholmod), Ptr{cholmod_factor}, (Csize_t, Ptr{cholmod_common}), n, Common)
end

function cholmod_l_allocate_factor(arg1, arg2)
    ccall((:cholmod_l_allocate_factor, libcholmod), Ptr{cholmod_factor}, (Csize_t, Ptr{cholmod_common}), arg1, arg2)
end

function cholmod_free_factor(L, Common)
    ccall((:cholmod_free_factor, libcholmod), Cint, (Ptr{Ptr{cholmod_factor}}, Ptr{cholmod_common}), L, Common)
end

function cholmod_l_free_factor(arg1, arg2)
    ccall((:cholmod_l_free_factor, libcholmod), Cint, (Ptr{Ptr{cholmod_factor}}, Ptr{cholmod_common}), arg1, arg2)
end

function cholmod_reallocate_factor(nznew, L, Common)
    ccall((:cholmod_reallocate_factor, libcholmod), Cint, (Csize_t, Ptr{cholmod_factor}, Ptr{cholmod_common}), nznew, L, Common)
end

function cholmod_l_reallocate_factor(arg1, arg2, arg3)
    ccall((:cholmod_l_reallocate_factor, libcholmod), Cint, (Csize_t, Ptr{cholmod_factor}, Ptr{cholmod_common}), arg1, arg2, arg3)
end

function cholmod_change_factor(to_xtype, to_ll, to_super, to_packed, to_monotonic, L, Common)
    ccall((:cholmod_change_factor, libcholmod), Cint, (Cint, Cint, Cint, Cint, Cint, Ptr{cholmod_factor}, Ptr{cholmod_common}), to_xtype, to_ll, to_super, to_packed, to_monotonic, L, Common)
end

function cholmod_l_change_factor(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    ccall((:cholmod_l_change_factor, libcholmod), Cint, (Cint, Cint, Cint, Cint, Cint, Ptr{cholmod_factor}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6, arg7)
end

function cholmod_pack_factor(L, Common)
    ccall((:cholmod_pack_factor, libcholmod), Cint, (Ptr{cholmod_factor}, Ptr{cholmod_common}), L, Common)
end

function cholmod_l_pack_factor(arg1, arg2)
    ccall((:cholmod_l_pack_factor, libcholmod), Cint, (Ptr{cholmod_factor}, Ptr{cholmod_common}), arg1, arg2)
end

function cholmod_reallocate_column(j, need, L, Common)
    ccall((:cholmod_reallocate_column, libcholmod), Cint, (Csize_t, Csize_t, Ptr{cholmod_factor}, Ptr{cholmod_common}), j, need, L, Common)
end

function cholmod_l_reallocate_column(arg1, arg2, arg3, arg4)
    ccall((:cholmod_l_reallocate_column, libcholmod), Cint, (Csize_t, Csize_t, Ptr{cholmod_factor}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4)
end

function cholmod_factor_to_sparse(L, Common)
    ccall((:cholmod_factor_to_sparse, libcholmod), Ptr{cholmod_sparse}, (Ptr{cholmod_factor}, Ptr{cholmod_common}), L, Common)
end

function cholmod_l_factor_to_sparse(arg1, arg2)
    ccall((:cholmod_l_factor_to_sparse, libcholmod), Ptr{cholmod_sparse}, (Ptr{cholmod_factor}, Ptr{cholmod_common}), arg1, arg2)
end

function cholmod_copy_factor(L, Common)
    ccall((:cholmod_copy_factor, libcholmod), Ptr{cholmod_factor}, (Ptr{cholmod_factor}, Ptr{cholmod_common}), L, Common)
end

function cholmod_l_copy_factor(arg1, arg2)
    ccall((:cholmod_l_copy_factor, libcholmod), Ptr{cholmod_factor}, (Ptr{cholmod_factor}, Ptr{cholmod_common}), arg1, arg2)
end

function cholmod_factor_xtype(to_xtype, L, Common)
    ccall((:cholmod_factor_xtype, libcholmod), Cint, (Cint, Ptr{cholmod_factor}, Ptr{cholmod_common}), to_xtype, L, Common)
end

function cholmod_l_factor_xtype(arg1, arg2, arg3)
    ccall((:cholmod_l_factor_xtype, libcholmod), Cint, (Cint, Ptr{cholmod_factor}, Ptr{cholmod_common}), arg1, arg2, arg3)
end

mutable struct cholmod_dense_struct
    nrow::Csize_t
    ncol::Csize_t
    nzmax::Csize_t
    d::Csize_t
    x::Ptr{Cvoid}
    z::Ptr{Cvoid}
    xtype::Cint
    dtype::Cint
    cholmod_dense_struct() = new()
end

const cholmod_dense = cholmod_dense_struct

function cholmod_allocate_dense(nrow, ncol, d, xtype, Common)
    ccall((:cholmod_allocate_dense, libcholmod), Ptr{cholmod_dense}, (Csize_t, Csize_t, Csize_t, Cint, Ptr{cholmod_common}), nrow, ncol, d, xtype, Common)
end

function cholmod_l_allocate_dense(arg1, arg2, arg3, arg4, arg5)
    ccall((:cholmod_l_allocate_dense, libcholmod), Ptr{cholmod_dense}, (Csize_t, Csize_t, Csize_t, Cint, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5)
end

function cholmod_zeros(nrow, ncol, xtype, Common)
    ccall((:cholmod_zeros, libcholmod), Ptr{cholmod_dense}, (Csize_t, Csize_t, Cint, Ptr{cholmod_common}), nrow, ncol, xtype, Common)
end

function cholmod_l_zeros(arg1, arg2, arg3, arg4)
    ccall((:cholmod_l_zeros, libcholmod), Ptr{cholmod_dense}, (Csize_t, Csize_t, Cint, Ptr{cholmod_common}), arg1, arg2, arg3, arg4)
end

function cholmod_ones(nrow, ncol, xtype, Common)
    ccall((:cholmod_ones, libcholmod), Ptr{cholmod_dense}, (Csize_t, Csize_t, Cint, Ptr{cholmod_common}), nrow, ncol, xtype, Common)
end

function cholmod_l_ones(arg1, arg2, arg3, arg4)
    ccall((:cholmod_l_ones, libcholmod), Ptr{cholmod_dense}, (Csize_t, Csize_t, Cint, Ptr{cholmod_common}), arg1, arg2, arg3, arg4)
end

function cholmod_eye(nrow, ncol, xtype, Common)
    ccall((:cholmod_eye, libcholmod), Ptr{cholmod_dense}, (Csize_t, Csize_t, Cint, Ptr{cholmod_common}), nrow, ncol, xtype, Common)
end

function cholmod_l_eye(arg1, arg2, arg3, arg4)
    ccall((:cholmod_l_eye, libcholmod), Ptr{cholmod_dense}, (Csize_t, Csize_t, Cint, Ptr{cholmod_common}), arg1, arg2, arg3, arg4)
end

function cholmod_free_dense(X, Common)
    ccall((:cholmod_free_dense, libcholmod), Cint, (Ptr{Ptr{cholmod_dense}}, Ptr{cholmod_common}), X, Common)
end

function cholmod_l_free_dense(arg1, arg2)
    ccall((:cholmod_l_free_dense, libcholmod), Cint, (Ptr{Ptr{cholmod_dense}}, Ptr{cholmod_common}), arg1, arg2)
end

function cholmod_ensure_dense(XHandle, nrow, ncol, d, xtype, Common)
    ccall((:cholmod_ensure_dense, libcholmod), Ptr{cholmod_dense}, (Ptr{Ptr{cholmod_dense}}, Csize_t, Csize_t, Csize_t, Cint, Ptr{cholmod_common}), XHandle, nrow, ncol, d, xtype, Common)
end

function cholmod_l_ensure_dense(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:cholmod_l_ensure_dense, libcholmod), Ptr{cholmod_dense}, (Ptr{Ptr{cholmod_dense}}, Csize_t, Csize_t, Csize_t, Cint, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6)
end

function cholmod_sparse_to_dense(A, Common)
    ccall((:cholmod_sparse_to_dense, libcholmod), Ptr{cholmod_dense}, (Ptr{cholmod_sparse}, Ptr{cholmod_common}), A, Common)
end

function cholmod_l_sparse_to_dense(arg1, arg2)
    ccall((:cholmod_l_sparse_to_dense, libcholmod), Ptr{cholmod_dense}, (Ptr{cholmod_sparse}, Ptr{cholmod_common}), arg1, arg2)
end

function cholmod_dense_to_sparse(X, values, Common)
    ccall((:cholmod_dense_to_sparse, libcholmod), Ptr{cholmod_sparse}, (Ptr{cholmod_dense}, Cint, Ptr{cholmod_common}), X, values, Common)
end

function cholmod_l_dense_to_sparse(arg1, arg2, arg3)
    ccall((:cholmod_l_dense_to_sparse, libcholmod), Ptr{cholmod_sparse}, (Ptr{cholmod_dense}, Cint, Ptr{cholmod_common}), arg1, arg2, arg3)
end

function cholmod_copy_dense(X, Common)
    ccall((:cholmod_copy_dense, libcholmod), Ptr{cholmod_dense}, (Ptr{cholmod_dense}, Ptr{cholmod_common}), X, Common)
end

function cholmod_l_copy_dense(arg1, arg2)
    ccall((:cholmod_l_copy_dense, libcholmod), Ptr{cholmod_dense}, (Ptr{cholmod_dense}, Ptr{cholmod_common}), arg1, arg2)
end

function cholmod_copy_dense2(X, Y, Common)
    ccall((:cholmod_copy_dense2, libcholmod), Cint, (Ptr{cholmod_dense}, Ptr{cholmod_dense}, Ptr{cholmod_common}), X, Y, Common)
end

function cholmod_l_copy_dense2(arg1, arg2, arg3)
    ccall((:cholmod_l_copy_dense2, libcholmod), Cint, (Ptr{cholmod_dense}, Ptr{cholmod_dense}, Ptr{cholmod_common}), arg1, arg2, arg3)
end

function cholmod_dense_xtype(to_xtype, X, Common)
    ccall((:cholmod_dense_xtype, libcholmod), Cint, (Cint, Ptr{cholmod_dense}, Ptr{cholmod_common}), to_xtype, X, Common)
end

function cholmod_l_dense_xtype(arg1, arg2, arg3)
    ccall((:cholmod_l_dense_xtype, libcholmod), Cint, (Cint, Ptr{cholmod_dense}, Ptr{cholmod_common}), arg1, arg2, arg3)
end

mutable struct cholmod_triplet_struct
    nrow::Csize_t
    ncol::Csize_t
    nzmax::Csize_t
    nnz::Csize_t
    i::Ptr{Cvoid}
    j::Ptr{Cvoid}
    x::Ptr{Cvoid}
    z::Ptr{Cvoid}
    stype::Cint
    itype::Cint
    xtype::Cint
    dtype::Cint
    cholmod_triplet_struct() = new()
end

const cholmod_triplet = cholmod_triplet_struct

function cholmod_allocate_triplet(nrow, ncol, nzmax, stype, xtype, Common)
    ccall((:cholmod_allocate_triplet, libcholmod), Ptr{cholmod_triplet}, (Csize_t, Csize_t, Csize_t, Cint, Cint, Ptr{cholmod_common}), nrow, ncol, nzmax, stype, xtype, Common)
end

function cholmod_l_allocate_triplet(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:cholmod_l_allocate_triplet, libcholmod), Ptr{cholmod_triplet}, (Csize_t, Csize_t, Csize_t, Cint, Cint, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6)
end

function cholmod_free_triplet(T, Common)
    ccall((:cholmod_free_triplet, libcholmod), Cint, (Ptr{Ptr{cholmod_triplet}}, Ptr{cholmod_common}), T, Common)
end

function cholmod_l_free_triplet(arg1, arg2)
    ccall((:cholmod_l_free_triplet, libcholmod), Cint, (Ptr{Ptr{cholmod_triplet}}, Ptr{cholmod_common}), arg1, arg2)
end

function cholmod_reallocate_triplet(nznew, T, Common)
    ccall((:cholmod_reallocate_triplet, libcholmod), Cint, (Csize_t, Ptr{cholmod_triplet}, Ptr{cholmod_common}), nznew, T, Common)
end

function cholmod_l_reallocate_triplet(arg1, arg2, arg3)
    ccall((:cholmod_l_reallocate_triplet, libcholmod), Cint, (Csize_t, Ptr{cholmod_triplet}, Ptr{cholmod_common}), arg1, arg2, arg3)
end

function cholmod_sparse_to_triplet(A, Common)
    ccall((:cholmod_sparse_to_triplet, libcholmod), Ptr{cholmod_triplet}, (Ptr{cholmod_sparse}, Ptr{cholmod_common}), A, Common)
end

function cholmod_l_sparse_to_triplet(arg1, arg2)
    ccall((:cholmod_l_sparse_to_triplet, libcholmod), Ptr{cholmod_triplet}, (Ptr{cholmod_sparse}, Ptr{cholmod_common}), arg1, arg2)
end

function cholmod_triplet_to_sparse(T, nzmax, Common)
    ccall((:cholmod_triplet_to_sparse, libcholmod), Ptr{cholmod_sparse}, (Ptr{cholmod_triplet}, Csize_t, Ptr{cholmod_common}), T, nzmax, Common)
end

function cholmod_l_triplet_to_sparse(arg1, arg2, arg3)
    ccall((:cholmod_l_triplet_to_sparse, libcholmod), Ptr{cholmod_sparse}, (Ptr{cholmod_triplet}, Csize_t, Ptr{cholmod_common}), arg1, arg2, arg3)
end

function cholmod_copy_triplet(T, Common)
    ccall((:cholmod_copy_triplet, libcholmod), Ptr{cholmod_triplet}, (Ptr{cholmod_triplet}, Ptr{cholmod_common}), T, Common)
end

function cholmod_l_copy_triplet(arg1, arg2)
    ccall((:cholmod_l_copy_triplet, libcholmod), Ptr{cholmod_triplet}, (Ptr{cholmod_triplet}, Ptr{cholmod_common}), arg1, arg2)
end

function cholmod_triplet_xtype(to_xtype, T, Common)
    ccall((:cholmod_triplet_xtype, libcholmod), Cint, (Cint, Ptr{cholmod_triplet}, Ptr{cholmod_common}), to_xtype, T, Common)
end

function cholmod_l_triplet_xtype(arg1, arg2, arg3)
    ccall((:cholmod_l_triplet_xtype, libcholmod), Cint, (Cint, Ptr{cholmod_triplet}, Ptr{cholmod_common}), arg1, arg2, arg3)
end

function cholmod_malloc(n, size, Common)
    ccall((:cholmod_malloc, libcholmod), Ptr{Cvoid}, (Csize_t, Csize_t, Ptr{cholmod_common}), n, size, Common)
end

function cholmod_l_malloc(arg1, arg2, arg3)
    ccall((:cholmod_l_malloc, libcholmod), Ptr{Cvoid}, (Csize_t, Csize_t, Ptr{cholmod_common}), arg1, arg2, arg3)
end

function cholmod_calloc(n, size, Common)
    ccall((:cholmod_calloc, libcholmod), Ptr{Cvoid}, (Csize_t, Csize_t, Ptr{cholmod_common}), n, size, Common)
end

function cholmod_l_calloc(arg1, arg2, arg3)
    ccall((:cholmod_l_calloc, libcholmod), Ptr{Cvoid}, (Csize_t, Csize_t, Ptr{cholmod_common}), arg1, arg2, arg3)
end

function cholmod_free(n, size, p, Common)
    ccall((:cholmod_free, libcholmod), Ptr{Cvoid}, (Csize_t, Csize_t, Ptr{Cvoid}, Ptr{cholmod_common}), n, size, p, Common)
end

function cholmod_l_free(arg1, arg2, arg3, arg4)
    ccall((:cholmod_l_free, libcholmod), Ptr{Cvoid}, (Csize_t, Csize_t, Ptr{Cvoid}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4)
end

function cholmod_realloc(nnew, size, p, n, Common)
    ccall((:cholmod_realloc, libcholmod), Ptr{Cvoid}, (Csize_t, Csize_t, Ptr{Cvoid}, Ptr{Csize_t}, Ptr{cholmod_common}), nnew, size, p, n, Common)
end

function cholmod_l_realloc(arg1, arg2, arg3, arg4, arg5)
    ccall((:cholmod_l_realloc, libcholmod), Ptr{Cvoid}, (Csize_t, Csize_t, Ptr{Cvoid}, Ptr{Csize_t}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5)
end

function cholmod_realloc_multiple(nnew, nint, xtype, Iblock, Jblock, Xblock, Zblock, n, Common)
    ccall((:cholmod_realloc_multiple, libcholmod), Cint, (Csize_t, Cint, Cint, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Ptr{cholmod_common}), nnew, nint, xtype, Iblock, Jblock, Xblock, Zblock, n, Common)
end

function cholmod_l_realloc_multiple(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)
    ccall((:cholmod_l_realloc_multiple, libcholmod), Cint, (Csize_t, Cint, Cint, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)
end

function cholmod_version(version)
    ccall((:cholmod_version, libcholmod), Cint, (Ptr{Cint},), version)
end

function cholmod_l_version(version)
    ccall((:cholmod_l_version, libcholmod), Cint, (Ptr{Cint},), version)
end

function cholmod_check_common(Common)
    ccall((:cholmod_check_common, libcholmod), Cint, (Ptr{cholmod_common},), Common)
end

function cholmod_l_check_common(arg1)
    ccall((:cholmod_l_check_common, libcholmod), Cint, (Ptr{cholmod_common},), arg1)
end

function cholmod_print_common(name, Common)
    ccall((:cholmod_print_common, libcholmod), Cint, (Ptr{Cchar}, Ptr{cholmod_common}), name, Common)
end

function cholmod_l_print_common(arg1, arg2)
    ccall((:cholmod_l_print_common, libcholmod), Cint, (Ptr{Cchar}, Ptr{cholmod_common}), arg1, arg2)
end

function cholmod_gpu_stats(arg1)
    ccall((:cholmod_gpu_stats, libcholmod), Cint, (Ptr{cholmod_common},), arg1)
end

function cholmod_l_gpu_stats(arg1)
    ccall((:cholmod_l_gpu_stats, libcholmod), Cint, (Ptr{cholmod_common},), arg1)
end

function cholmod_check_sparse(A, Common)
    ccall((:cholmod_check_sparse, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{cholmod_common}), A, Common)
end

function cholmod_l_check_sparse(arg1, arg2)
    ccall((:cholmod_l_check_sparse, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{cholmod_common}), arg1, arg2)
end

function cholmod_print_sparse(A, name, Common)
    ccall((:cholmod_print_sparse, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{Cchar}, Ptr{cholmod_common}), A, name, Common)
end

function cholmod_l_print_sparse(arg1, arg2, arg3)
    ccall((:cholmod_l_print_sparse, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{Cchar}, Ptr{cholmod_common}), arg1, arg2, arg3)
end

function cholmod_check_dense(X, Common)
    ccall((:cholmod_check_dense, libcholmod), Cint, (Ptr{cholmod_dense}, Ptr{cholmod_common}), X, Common)
end

function cholmod_l_check_dense(arg1, arg2)
    ccall((:cholmod_l_check_dense, libcholmod), Cint, (Ptr{cholmod_dense}, Ptr{cholmod_common}), arg1, arg2)
end

function cholmod_print_dense(X, name, Common)
    ccall((:cholmod_print_dense, libcholmod), Cint, (Ptr{cholmod_dense}, Ptr{Cchar}, Ptr{cholmod_common}), X, name, Common)
end

function cholmod_l_print_dense(arg1, arg2, arg3)
    ccall((:cholmod_l_print_dense, libcholmod), Cint, (Ptr{cholmod_dense}, Ptr{Cchar}, Ptr{cholmod_common}), arg1, arg2, arg3)
end

function cholmod_check_factor(L, Common)
    ccall((:cholmod_check_factor, libcholmod), Cint, (Ptr{cholmod_factor}, Ptr{cholmod_common}), L, Common)
end

function cholmod_l_check_factor(arg1, arg2)
    ccall((:cholmod_l_check_factor, libcholmod), Cint, (Ptr{cholmod_factor}, Ptr{cholmod_common}), arg1, arg2)
end

function cholmod_print_factor(L, name, Common)
    ccall((:cholmod_print_factor, libcholmod), Cint, (Ptr{cholmod_factor}, Ptr{Cchar}, Ptr{cholmod_common}), L, name, Common)
end

function cholmod_l_print_factor(arg1, arg2, arg3)
    ccall((:cholmod_l_print_factor, libcholmod), Cint, (Ptr{cholmod_factor}, Ptr{Cchar}, Ptr{cholmod_common}), arg1, arg2, arg3)
end

function cholmod_check_triplet(T, Common)
    ccall((:cholmod_check_triplet, libcholmod), Cint, (Ptr{cholmod_triplet}, Ptr{cholmod_common}), T, Common)
end

function cholmod_l_check_triplet(arg1, arg2)
    ccall((:cholmod_l_check_triplet, libcholmod), Cint, (Ptr{cholmod_triplet}, Ptr{cholmod_common}), arg1, arg2)
end

function cholmod_print_triplet(T, name, Common)
    ccall((:cholmod_print_triplet, libcholmod), Cint, (Ptr{cholmod_triplet}, Ptr{Cchar}, Ptr{cholmod_common}), T, name, Common)
end

function cholmod_l_print_triplet(arg1, arg2, arg3)
    ccall((:cholmod_l_print_triplet, libcholmod), Cint, (Ptr{cholmod_triplet}, Ptr{Cchar}, Ptr{cholmod_common}), arg1, arg2, arg3)
end

function cholmod_check_subset(Set, len, n, Common)
    ccall((:cholmod_check_subset, libcholmod), Cint, (Ptr{Int32}, Int64, Csize_t, Ptr{cholmod_common}), Set, len, n, Common)
end

function cholmod_l_check_subset(arg1, arg2, arg3, arg4)
    ccall((:cholmod_l_check_subset, libcholmod), Cint, (Ptr{Int64}, Int64, Csize_t, Ptr{cholmod_common}), arg1, arg2, arg3, arg4)
end

function cholmod_print_subset(Set, len, n, name, Common)
    ccall((:cholmod_print_subset, libcholmod), Cint, (Ptr{Int32}, Int64, Csize_t, Ptr{Cchar}, Ptr{cholmod_common}), Set, len, n, name, Common)
end

function cholmod_l_print_subset(arg1, arg2, arg3, arg4, arg5)
    ccall((:cholmod_l_print_subset, libcholmod), Cint, (Ptr{Int64}, Int64, Csize_t, Ptr{Cchar}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5)
end

function cholmod_check_perm(Perm, len, n, Common)
    ccall((:cholmod_check_perm, libcholmod), Cint, (Ptr{Int32}, Csize_t, Csize_t, Ptr{cholmod_common}), Perm, len, n, Common)
end

function cholmod_l_check_perm(arg1, arg2, arg3, arg4)
    ccall((:cholmod_l_check_perm, libcholmod), Cint, (Ptr{Int64}, Csize_t, Csize_t, Ptr{cholmod_common}), arg1, arg2, arg3, arg4)
end

function cholmod_print_perm(Perm, len, n, name, Common)
    ccall((:cholmod_print_perm, libcholmod), Cint, (Ptr{Int32}, Csize_t, Csize_t, Ptr{Cchar}, Ptr{cholmod_common}), Perm, len, n, name, Common)
end

function cholmod_l_print_perm(arg1, arg2, arg3, arg4, arg5)
    ccall((:cholmod_l_print_perm, libcholmod), Cint, (Ptr{Int64}, Csize_t, Csize_t, Ptr{Cchar}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5)
end

function cholmod_check_parent(Parent, n, Common)
    ccall((:cholmod_check_parent, libcholmod), Cint, (Ptr{Int32}, Csize_t, Ptr{cholmod_common}), Parent, n, Common)
end

function cholmod_l_check_parent(arg1, arg2, arg3)
    ccall((:cholmod_l_check_parent, libcholmod), Cint, (Ptr{Int64}, Csize_t, Ptr{cholmod_common}), arg1, arg2, arg3)
end

function cholmod_print_parent(Parent, n, name, Common)
    ccall((:cholmod_print_parent, libcholmod), Cint, (Ptr{Int32}, Csize_t, Ptr{Cchar}, Ptr{cholmod_common}), Parent, n, name, Common)
end

function cholmod_l_print_parent(arg1, arg2, arg3, arg4)
    ccall((:cholmod_l_print_parent, libcholmod), Cint, (Ptr{Int64}, Csize_t, Ptr{Cchar}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4)
end

function cholmod_read_sparse(f, Common)
    ccall((:cholmod_read_sparse, libcholmod), Ptr{cholmod_sparse}, (Ptr{Libc.FILE}, Ptr{cholmod_common}), f, Common)
end

function cholmod_l_read_sparse(arg1, arg2)
    ccall((:cholmod_l_read_sparse, libcholmod), Ptr{cholmod_sparse}, (Ptr{Libc.FILE}, Ptr{cholmod_common}), arg1, arg2)
end

function cholmod_read_triplet(f, Common)
    ccall((:cholmod_read_triplet, libcholmod), Ptr{cholmod_triplet}, (Ptr{Libc.FILE}, Ptr{cholmod_common}), f, Common)
end

function cholmod_l_read_triplet(arg1, arg2)
    ccall((:cholmod_l_read_triplet, libcholmod), Ptr{cholmod_triplet}, (Ptr{Libc.FILE}, Ptr{cholmod_common}), arg1, arg2)
end

function cholmod_read_dense(f, Common)
    ccall((:cholmod_read_dense, libcholmod), Ptr{cholmod_dense}, (Ptr{Libc.FILE}, Ptr{cholmod_common}), f, Common)
end

function cholmod_l_read_dense(arg1, arg2)
    ccall((:cholmod_l_read_dense, libcholmod), Ptr{cholmod_dense}, (Ptr{Libc.FILE}, Ptr{cholmod_common}), arg1, arg2)
end

function cholmod_read_matrix(f, prefer, mtype, Common)
    ccall((:cholmod_read_matrix, libcholmod), Ptr{Cvoid}, (Ptr{Libc.FILE}, Cint, Ptr{Cint}, Ptr{cholmod_common}), f, prefer, mtype, Common)
end

function cholmod_l_read_matrix(arg1, arg2, arg3, arg4)
    ccall((:cholmod_l_read_matrix, libcholmod), Ptr{Cvoid}, (Ptr{Libc.FILE}, Cint, Ptr{Cint}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4)
end

function cholmod_write_sparse(f, A, Z, comments, Common)
    ccall((:cholmod_write_sparse, libcholmod), Cint, (Ptr{Libc.FILE}, Ptr{cholmod_sparse}, Ptr{cholmod_sparse}, Ptr{Cchar}, Ptr{cholmod_common}), f, A, Z, comments, Common)
end

function cholmod_l_write_sparse(arg1, arg2, arg3, c, arg5)
    ccall((:cholmod_l_write_sparse, libcholmod), Cint, (Ptr{Libc.FILE}, Ptr{cholmod_sparse}, Ptr{cholmod_sparse}, Ptr{Cchar}, Ptr{cholmod_common}), arg1, arg2, arg3, c, arg5)
end

function cholmod_write_dense(f, X, comments, Common)
    ccall((:cholmod_write_dense, libcholmod), Cint, (Ptr{Libc.FILE}, Ptr{cholmod_dense}, Ptr{Cchar}, Ptr{cholmod_common}), f, X, comments, Common)
end

function cholmod_l_write_dense(arg1, arg2, arg3, arg4)
    ccall((:cholmod_l_write_dense, libcholmod), Cint, (Ptr{Libc.FILE}, Ptr{cholmod_dense}, Ptr{Cchar}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4)
end

function cholmod_analyze(A, Common)
    ccall((:cholmod_analyze, libcholmod), Ptr{cholmod_factor}, (Ptr{cholmod_sparse}, Ptr{cholmod_common}), A, Common)
end

function cholmod_l_analyze(arg1, arg2)
    ccall((:cholmod_l_analyze, libcholmod), Ptr{cholmod_factor}, (Ptr{cholmod_sparse}, Ptr{cholmod_common}), arg1, arg2)
end

function cholmod_analyze_p(A, UserPerm, fset, fsize, Common)
    ccall((:cholmod_analyze_p, libcholmod), Ptr{cholmod_factor}, (Ptr{cholmod_sparse}, Ptr{Int32}, Ptr{Int32}, Csize_t, Ptr{cholmod_common}), A, UserPerm, fset, fsize, Common)
end

function cholmod_l_analyze_p(arg1, arg2, arg3, arg4, arg5)
    ccall((:cholmod_l_analyze_p, libcholmod), Ptr{cholmod_factor}, (Ptr{cholmod_sparse}, Ptr{Int64}, Ptr{Int64}, Csize_t, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5)
end

function cholmod_analyze_p2(for_whom, A, UserPerm, fset, fsize, Common)
    ccall((:cholmod_analyze_p2, libcholmod), Ptr{cholmod_factor}, (Cint, Ptr{cholmod_sparse}, Ptr{Int32}, Ptr{Int32}, Csize_t, Ptr{cholmod_common}), for_whom, A, UserPerm, fset, fsize, Common)
end

function cholmod_l_analyze_p2(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:cholmod_l_analyze_p2, libcholmod), Ptr{cholmod_factor}, (Cint, Ptr{cholmod_sparse}, Ptr{Int64}, Ptr{Int64}, Csize_t, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6)
end

function cholmod_factorize(A, L, Common)
    ccall((:cholmod_factorize, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{cholmod_factor}, Ptr{cholmod_common}), A, L, Common)
end

function cholmod_l_factorize(arg1, arg2, arg3)
    ccall((:cholmod_l_factorize, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{cholmod_factor}, Ptr{cholmod_common}), arg1, arg2, arg3)
end

function cholmod_factorize_p(A, beta, fset, fsize, L, Common)
    ccall((:cholmod_factorize_p, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{Cdouble}, Ptr{Int32}, Csize_t, Ptr{cholmod_factor}, Ptr{cholmod_common}), A, beta, fset, fsize, L, Common)
end

function cholmod_l_factorize_p(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:cholmod_l_factorize_p, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{Cdouble}, Ptr{Int64}, Csize_t, Ptr{cholmod_factor}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6)
end

function cholmod_solve(sys, L, B, Common)
    ccall((:cholmod_solve, libcholmod), Ptr{cholmod_dense}, (Cint, Ptr{cholmod_factor}, Ptr{cholmod_dense}, Ptr{cholmod_common}), sys, L, B, Common)
end

function cholmod_l_solve(arg1, arg2, arg3, arg4)
    ccall((:cholmod_l_solve, libcholmod), Ptr{cholmod_dense}, (Cint, Ptr{cholmod_factor}, Ptr{cholmod_dense}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4)
end

function cholmod_solve2(sys, L, B, Bset, X_Handle, Xset_Handle, Y_Handle, E_Handle, Common)
    ccall((:cholmod_solve2, libcholmod), Cint, (Cint, Ptr{cholmod_factor}, Ptr{cholmod_dense}, Ptr{cholmod_sparse}, Ptr{Ptr{cholmod_dense}}, Ptr{Ptr{cholmod_sparse}}, Ptr{Ptr{cholmod_dense}}, Ptr{Ptr{cholmod_dense}}, Ptr{cholmod_common}), sys, L, B, Bset, X_Handle, Xset_Handle, Y_Handle, E_Handle, Common)
end

function cholmod_l_solve2(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)
    ccall((:cholmod_l_solve2, libcholmod), Cint, (Cint, Ptr{cholmod_factor}, Ptr{cholmod_dense}, Ptr{cholmod_sparse}, Ptr{Ptr{cholmod_dense}}, Ptr{Ptr{cholmod_sparse}}, Ptr{Ptr{cholmod_dense}}, Ptr{Ptr{cholmod_dense}}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)
end

function cholmod_spsolve(sys, L, B, Common)
    ccall((:cholmod_spsolve, libcholmod), Ptr{cholmod_sparse}, (Cint, Ptr{cholmod_factor}, Ptr{cholmod_sparse}, Ptr{cholmod_common}), sys, L, B, Common)
end

function cholmod_l_spsolve(arg1, arg2, arg3, arg4)
    ccall((:cholmod_l_spsolve, libcholmod), Ptr{cholmod_sparse}, (Cint, Ptr{cholmod_factor}, Ptr{cholmod_sparse}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4)
end

function cholmod_etree(A, Parent, Common)
    ccall((:cholmod_etree, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{Int32}, Ptr{cholmod_common}), A, Parent, Common)
end

function cholmod_l_etree(arg1, arg2, arg3)
    ccall((:cholmod_l_etree, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{Int64}, Ptr{cholmod_common}), arg1, arg2, arg3)
end

function cholmod_rowcolcounts(A, fset, fsize, Parent, Post, RowCount, ColCount, First, Level, Common)
    ccall((:cholmod_rowcolcounts, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{Int32}, Csize_t, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{cholmod_common}), A, fset, fsize, Parent, Post, RowCount, ColCount, First, Level, Common)
end

function cholmod_l_rowcolcounts(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)
    ccall((:cholmod_l_rowcolcounts, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{Int64}, Csize_t, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)
end

function cholmod_analyze_ordering(A, ordering, Perm, fset, fsize, Parent, Post, ColCount, First, Level, Common)
    ccall((:cholmod_analyze_ordering, libcholmod), Cint, (Ptr{cholmod_sparse}, Cint, Ptr{Int32}, Ptr{Int32}, Csize_t, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{cholmod_common}), A, ordering, Perm, fset, fsize, Parent, Post, ColCount, First, Level, Common)
end

function cholmod_l_analyze_ordering(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11)
    ccall((:cholmod_l_analyze_ordering, libcholmod), Cint, (Ptr{cholmod_sparse}, Cint, Ptr{Int64}, Ptr{Int64}, Csize_t, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11)
end

function cholmod_amd(A, fset, fsize, Perm, Common)
    ccall((:cholmod_amd, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{Int32}, Csize_t, Ptr{Int32}, Ptr{cholmod_common}), A, fset, fsize, Perm, Common)
end

function cholmod_l_amd(arg1, arg2, arg3, arg4, arg5)
    ccall((:cholmod_l_amd, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{Int64}, Csize_t, Ptr{Int64}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5)
end

function cholmod_colamd(A, fset, fsize, postorder, Perm, Common)
    ccall((:cholmod_colamd, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{Int32}, Csize_t, Cint, Ptr{Int32}, Ptr{cholmod_common}), A, fset, fsize, postorder, Perm, Common)
end

function cholmod_l_colamd(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:cholmod_l_colamd, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{Int64}, Csize_t, Cint, Ptr{Int64}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6)
end

function cholmod_rowfac(A, F, beta, kstart, kend, L, Common)
    ccall((:cholmod_rowfac, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{cholmod_sparse}, Ptr{Cdouble}, Csize_t, Csize_t, Ptr{cholmod_factor}, Ptr{cholmod_common}), A, F, beta, kstart, kend, L, Common)
end

function cholmod_l_rowfac(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    ccall((:cholmod_l_rowfac, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{cholmod_sparse}, Ptr{Cdouble}, Csize_t, Csize_t, Ptr{cholmod_factor}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6, arg7)
end

function cholmod_rowfac_mask(A, F, beta, kstart, kend, mask, RLinkUp, L, Common)
    ccall((:cholmod_rowfac_mask, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{cholmod_sparse}, Ptr{Cdouble}, Csize_t, Csize_t, Ptr{Int32}, Ptr{Int32}, Ptr{cholmod_factor}, Ptr{cholmod_common}), A, F, beta, kstart, kend, mask, RLinkUp, L, Common)
end

function cholmod_l_rowfac_mask(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)
    ccall((:cholmod_l_rowfac_mask, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{cholmod_sparse}, Ptr{Cdouble}, Csize_t, Csize_t, Ptr{Int64}, Ptr{Int64}, Ptr{cholmod_factor}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)
end

function cholmod_rowfac_mask2(A, F, beta, kstart, kend, mask, maskmark, RLinkUp, L, Common)
    ccall((:cholmod_rowfac_mask2, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{cholmod_sparse}, Ptr{Cdouble}, Csize_t, Csize_t, Ptr{Int32}, Int32, Ptr{Int32}, Ptr{cholmod_factor}, Ptr{cholmod_common}), A, F, beta, kstart, kend, mask, maskmark, RLinkUp, L, Common)
end

function cholmod_l_rowfac_mask2(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)
    ccall((:cholmod_l_rowfac_mask2, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{cholmod_sparse}, Ptr{Cdouble}, Csize_t, Csize_t, Ptr{Int64}, Int64, Ptr{Int64}, Ptr{cholmod_factor}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)
end

function cholmod_row_subtree(A, F, k, Parent, R, Common)
    ccall((:cholmod_row_subtree, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{cholmod_sparse}, Csize_t, Ptr{Int32}, Ptr{cholmod_sparse}, Ptr{cholmod_common}), A, F, k, Parent, R, Common)
end

function cholmod_l_row_subtree(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:cholmod_l_row_subtree, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{cholmod_sparse}, Csize_t, Ptr{Int64}, Ptr{cholmod_sparse}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6)
end

function cholmod_lsolve_pattern(B, L, X, Common)
    ccall((:cholmod_lsolve_pattern, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{cholmod_factor}, Ptr{cholmod_sparse}, Ptr{cholmod_common}), B, L, X, Common)
end

function cholmod_l_lsolve_pattern(arg1, arg2, arg3, arg4)
    ccall((:cholmod_l_lsolve_pattern, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{cholmod_factor}, Ptr{cholmod_sparse}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4)
end

function cholmod_row_lsubtree(A, Fi, fnz, k, L, R, Common)
    ccall((:cholmod_row_lsubtree, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{Int32}, Csize_t, Csize_t, Ptr{cholmod_factor}, Ptr{cholmod_sparse}, Ptr{cholmod_common}), A, Fi, fnz, k, L, R, Common)
end

function cholmod_l_row_lsubtree(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    ccall((:cholmod_l_row_lsubtree, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{Int64}, Csize_t, Csize_t, Ptr{cholmod_factor}, Ptr{cholmod_sparse}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6, arg7)
end

function cholmod_resymbol(A, fset, fsize, pack, L, Common)
    ccall((:cholmod_resymbol, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{Int32}, Csize_t, Cint, Ptr{cholmod_factor}, Ptr{cholmod_common}), A, fset, fsize, pack, L, Common)
end

function cholmod_l_resymbol(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:cholmod_l_resymbol, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{Int64}, Csize_t, Cint, Ptr{cholmod_factor}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6)
end

function cholmod_resymbol_noperm(A, fset, fsize, pack, L, Common)
    ccall((:cholmod_resymbol_noperm, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{Int32}, Csize_t, Cint, Ptr{cholmod_factor}, Ptr{cholmod_common}), A, fset, fsize, pack, L, Common)
end

function cholmod_l_resymbol_noperm(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:cholmod_l_resymbol_noperm, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{Int64}, Csize_t, Cint, Ptr{cholmod_factor}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6)
end

function cholmod_rcond(L, Common)
    ccall((:cholmod_rcond, libcholmod), Cdouble, (Ptr{cholmod_factor}, Ptr{cholmod_common}), L, Common)
end

function cholmod_l_rcond(arg1, arg2)
    ccall((:cholmod_l_rcond, libcholmod), Cdouble, (Ptr{cholmod_factor}, Ptr{cholmod_common}), arg1, arg2)
end

function cholmod_postorder(Parent, n, Weight_p, Post, Common)
    ccall((:cholmod_postorder, libcholmod), Int32, (Ptr{Int32}, Csize_t, Ptr{Int32}, Ptr{Int32}, Ptr{cholmod_common}), Parent, n, Weight_p, Post, Common)
end

function cholmod_l_postorder(arg1, arg2, arg3, arg4, arg5)
    ccall((:cholmod_l_postorder, libcholmod), Int64, (Ptr{Int64}, Csize_t, Ptr{Int64}, Ptr{Int64}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5)
end

function cholmod_drop(tol, A, Common)
    ccall((:cholmod_drop, libcholmod), Cint, (Cdouble, Ptr{cholmod_sparse}, Ptr{cholmod_common}), tol, A, Common)
end

function cholmod_l_drop(arg1, arg2, arg3)
    ccall((:cholmod_l_drop, libcholmod), Cint, (Cdouble, Ptr{cholmod_sparse}, Ptr{cholmod_common}), arg1, arg2, arg3)
end

function cholmod_norm_dense(X, norm, Common)
    ccall((:cholmod_norm_dense, libcholmod), Cdouble, (Ptr{cholmod_dense}, Cint, Ptr{cholmod_common}), X, norm, Common)
end

function cholmod_l_norm_dense(arg1, arg2, arg3)
    ccall((:cholmod_l_norm_dense, libcholmod), Cdouble, (Ptr{cholmod_dense}, Cint, Ptr{cholmod_common}), arg1, arg2, arg3)
end

function cholmod_norm_sparse(A, norm, Common)
    ccall((:cholmod_norm_sparse, libcholmod), Cdouble, (Ptr{cholmod_sparse}, Cint, Ptr{cholmod_common}), A, norm, Common)
end

function cholmod_l_norm_sparse(arg1, arg2, arg3)
    ccall((:cholmod_l_norm_sparse, libcholmod), Cdouble, (Ptr{cholmod_sparse}, Cint, Ptr{cholmod_common}), arg1, arg2, arg3)
end

function cholmod_horzcat(A, B, values, Common)
    ccall((:cholmod_horzcat, libcholmod), Ptr{cholmod_sparse}, (Ptr{cholmod_sparse}, Ptr{cholmod_sparse}, Cint, Ptr{cholmod_common}), A, B, values, Common)
end

function cholmod_l_horzcat(arg1, arg2, arg3, arg4)
    ccall((:cholmod_l_horzcat, libcholmod), Ptr{cholmod_sparse}, (Ptr{cholmod_sparse}, Ptr{cholmod_sparse}, Cint, Ptr{cholmod_common}), arg1, arg2, arg3, arg4)
end

function cholmod_scale(S, scale, A, Common)
    ccall((:cholmod_scale, libcholmod), Cint, (Ptr{cholmod_dense}, Cint, Ptr{cholmod_sparse}, Ptr{cholmod_common}), S, scale, A, Common)
end

function cholmod_l_scale(arg1, arg2, arg3, arg4)
    ccall((:cholmod_l_scale, libcholmod), Cint, (Ptr{cholmod_dense}, Cint, Ptr{cholmod_sparse}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4)
end

function cholmod_sdmult(A, transpose, alpha, beta, X, Y, Common)
    ccall((:cholmod_sdmult, libcholmod), Cint, (Ptr{cholmod_sparse}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{cholmod_dense}, Ptr{cholmod_dense}, Ptr{cholmod_common}), A, transpose, alpha, beta, X, Y, Common)
end

function cholmod_l_sdmult(arg1, arg2, arg3, arg4, arg5, Y, arg7)
    ccall((:cholmod_l_sdmult, libcholmod), Cint, (Ptr{cholmod_sparse}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{cholmod_dense}, Ptr{cholmod_dense}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, Y, arg7)
end

function cholmod_ssmult(A, B, stype, values, sorted, Common)
    ccall((:cholmod_ssmult, libcholmod), Ptr{cholmod_sparse}, (Ptr{cholmod_sparse}, Ptr{cholmod_sparse}, Cint, Cint, Cint, Ptr{cholmod_common}), A, B, stype, values, sorted, Common)
end

function cholmod_l_ssmult(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:cholmod_l_ssmult, libcholmod), Ptr{cholmod_sparse}, (Ptr{cholmod_sparse}, Ptr{cholmod_sparse}, Cint, Cint, Cint, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6)
end

function cholmod_submatrix(A, rset, rsize, cset, csize, values, sorted, Common)
    ccall((:cholmod_submatrix, libcholmod), Ptr{cholmod_sparse}, (Ptr{cholmod_sparse}, Ptr{Int32}, Int64, Ptr{Int32}, Int64, Cint, Cint, Ptr{cholmod_common}), A, rset, rsize, cset, csize, values, sorted, Common)
end

function cholmod_l_submatrix(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
    ccall((:cholmod_l_submatrix, libcholmod), Ptr{cholmod_sparse}, (Ptr{cholmod_sparse}, Ptr{Int64}, Int64, Ptr{Int64}, Int64, Cint, Cint, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
end

function cholmod_vertcat(A, B, values, Common)
    ccall((:cholmod_vertcat, libcholmod), Ptr{cholmod_sparse}, (Ptr{cholmod_sparse}, Ptr{cholmod_sparse}, Cint, Ptr{cholmod_common}), A, B, values, Common)
end

function cholmod_l_vertcat(arg1, arg2, arg3, arg4)
    ccall((:cholmod_l_vertcat, libcholmod), Ptr{cholmod_sparse}, (Ptr{cholmod_sparse}, Ptr{cholmod_sparse}, Cint, Ptr{cholmod_common}), arg1, arg2, arg3, arg4)
end

function cholmod_symmetry(A, option, xmatched, pmatched, nzoffdiag, nzdiag, Common)
    ccall((:cholmod_symmetry, libcholmod), Cint, (Ptr{cholmod_sparse}, Cint, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{cholmod_common}), A, option, xmatched, pmatched, nzoffdiag, nzdiag, Common)
end

function cholmod_l_symmetry(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    ccall((:cholmod_l_symmetry, libcholmod), Cint, (Ptr{cholmod_sparse}, Cint, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6, arg7)
end

function cholmod_updown(update, C, L, Common)
    ccall((:cholmod_updown, libcholmod), Cint, (Cint, Ptr{cholmod_sparse}, Ptr{cholmod_factor}, Ptr{cholmod_common}), update, C, L, Common)
end

function cholmod_l_updown(arg1, arg2, arg3, arg4)
    ccall((:cholmod_l_updown, libcholmod), Cint, (Cint, Ptr{cholmod_sparse}, Ptr{cholmod_factor}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4)
end

function cholmod_updown_solve(update, C, L, X, DeltaB, Common)
    ccall((:cholmod_updown_solve, libcholmod), Cint, (Cint, Ptr{cholmod_sparse}, Ptr{cholmod_factor}, Ptr{cholmod_dense}, Ptr{cholmod_dense}, Ptr{cholmod_common}), update, C, L, X, DeltaB, Common)
end

function cholmod_l_updown_solve(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:cholmod_l_updown_solve, libcholmod), Cint, (Cint, Ptr{cholmod_sparse}, Ptr{cholmod_factor}, Ptr{cholmod_dense}, Ptr{cholmod_dense}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6)
end

function cholmod_updown_mark(update, C, colmark, L, X, DeltaB, Common)
    ccall((:cholmod_updown_mark, libcholmod), Cint, (Cint, Ptr{cholmod_sparse}, Ptr{Int32}, Ptr{cholmod_factor}, Ptr{cholmod_dense}, Ptr{cholmod_dense}, Ptr{cholmod_common}), update, C, colmark, L, X, DeltaB, Common)
end

function cholmod_l_updown_mark(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    ccall((:cholmod_l_updown_mark, libcholmod), Cint, (Cint, Ptr{cholmod_sparse}, Ptr{Int64}, Ptr{cholmod_factor}, Ptr{cholmod_dense}, Ptr{cholmod_dense}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6, arg7)
end

function cholmod_updown_mask(update, C, colmark, mask, L, X, DeltaB, Common)
    ccall((:cholmod_updown_mask, libcholmod), Cint, (Cint, Ptr{cholmod_sparse}, Ptr{Int32}, Ptr{Int32}, Ptr{cholmod_factor}, Ptr{cholmod_dense}, Ptr{cholmod_dense}, Ptr{cholmod_common}), update, C, colmark, mask, L, X, DeltaB, Common)
end

function cholmod_l_updown_mask(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
    ccall((:cholmod_l_updown_mask, libcholmod), Cint, (Cint, Ptr{cholmod_sparse}, Ptr{Int64}, Ptr{Int64}, Ptr{cholmod_factor}, Ptr{cholmod_dense}, Ptr{cholmod_dense}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
end

function cholmod_updown_mask2(update, C, colmark, mask, maskmark, L, X, DeltaB, Common)
    ccall((:cholmod_updown_mask2, libcholmod), Cint, (Cint, Ptr{cholmod_sparse}, Ptr{Int32}, Ptr{Int32}, Int32, Ptr{cholmod_factor}, Ptr{cholmod_dense}, Ptr{cholmod_dense}, Ptr{cholmod_common}), update, C, colmark, mask, maskmark, L, X, DeltaB, Common)
end

function cholmod_l_updown_mask2(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)
    ccall((:cholmod_l_updown_mask2, libcholmod), Cint, (Cint, Ptr{cholmod_sparse}, Ptr{Int64}, Ptr{Int64}, Int64, Ptr{cholmod_factor}, Ptr{cholmod_dense}, Ptr{cholmod_dense}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)
end

function cholmod_rowadd(k, R, L, Common)
    ccall((:cholmod_rowadd, libcholmod), Cint, (Csize_t, Ptr{cholmod_sparse}, Ptr{cholmod_factor}, Ptr{cholmod_common}), k, R, L, Common)
end

function cholmod_l_rowadd(arg1, arg2, arg3, arg4)
    ccall((:cholmod_l_rowadd, libcholmod), Cint, (Csize_t, Ptr{cholmod_sparse}, Ptr{cholmod_factor}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4)
end

function cholmod_rowadd_solve(k, R, bk, L, X, DeltaB, Common)
    ccall((:cholmod_rowadd_solve, libcholmod), Cint, (Csize_t, Ptr{cholmod_sparse}, Ptr{Cdouble}, Ptr{cholmod_factor}, Ptr{cholmod_dense}, Ptr{cholmod_dense}, Ptr{cholmod_common}), k, R, bk, L, X, DeltaB, Common)
end

function cholmod_l_rowadd_solve(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    ccall((:cholmod_l_rowadd_solve, libcholmod), Cint, (Csize_t, Ptr{cholmod_sparse}, Ptr{Cdouble}, Ptr{cholmod_factor}, Ptr{cholmod_dense}, Ptr{cholmod_dense}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6, arg7)
end

function cholmod_rowadd_mark(k, R, bk, colmark, L, X, DeltaB, Common)
    ccall((:cholmod_rowadd_mark, libcholmod), Cint, (Csize_t, Ptr{cholmod_sparse}, Ptr{Cdouble}, Ptr{Int32}, Ptr{cholmod_factor}, Ptr{cholmod_dense}, Ptr{cholmod_dense}, Ptr{cholmod_common}), k, R, bk, colmark, L, X, DeltaB, Common)
end

function cholmod_l_rowadd_mark(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
    ccall((:cholmod_l_rowadd_mark, libcholmod), Cint, (Csize_t, Ptr{cholmod_sparse}, Ptr{Cdouble}, Ptr{Int64}, Ptr{cholmod_factor}, Ptr{cholmod_dense}, Ptr{cholmod_dense}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
end

function cholmod_rowdel(k, R, L, Common)
    ccall((:cholmod_rowdel, libcholmod), Cint, (Csize_t, Ptr{cholmod_sparse}, Ptr{cholmod_factor}, Ptr{cholmod_common}), k, R, L, Common)
end

function cholmod_l_rowdel(arg1, arg2, arg3, arg4)
    ccall((:cholmod_l_rowdel, libcholmod), Cint, (Csize_t, Ptr{cholmod_sparse}, Ptr{cholmod_factor}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4)
end

function cholmod_rowdel_solve(k, R, yk, L, X, DeltaB, Common)
    ccall((:cholmod_rowdel_solve, libcholmod), Cint, (Csize_t, Ptr{cholmod_sparse}, Ptr{Cdouble}, Ptr{cholmod_factor}, Ptr{cholmod_dense}, Ptr{cholmod_dense}, Ptr{cholmod_common}), k, R, yk, L, X, DeltaB, Common)
end

function cholmod_l_rowdel_solve(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    ccall((:cholmod_l_rowdel_solve, libcholmod), Cint, (Csize_t, Ptr{cholmod_sparse}, Ptr{Cdouble}, Ptr{cholmod_factor}, Ptr{cholmod_dense}, Ptr{cholmod_dense}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6, arg7)
end

function cholmod_rowdel_mark(k, R, yk, colmark, L, X, DeltaB, Common)
    ccall((:cholmod_rowdel_mark, libcholmod), Cint, (Csize_t, Ptr{cholmod_sparse}, Ptr{Cdouble}, Ptr{Int32}, Ptr{cholmod_factor}, Ptr{cholmod_dense}, Ptr{cholmod_dense}, Ptr{cholmod_common}), k, R, yk, colmark, L, X, DeltaB, Common)
end

function cholmod_l_rowdel_mark(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
    ccall((:cholmod_l_rowdel_mark, libcholmod), Cint, (Csize_t, Ptr{cholmod_sparse}, Ptr{Cdouble}, Ptr{Int64}, Ptr{cholmod_factor}, Ptr{cholmod_dense}, Ptr{cholmod_dense}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
end

function cholmod_ccolamd(A, fset, fsize, Cmember, Perm, Common)
    ccall((:cholmod_ccolamd, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{Int32}, Csize_t, Ptr{Int32}, Ptr{Int32}, Ptr{cholmod_common}), A, fset, fsize, Cmember, Perm, Common)
end

function cholmod_l_ccolamd(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:cholmod_l_ccolamd, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{Int64}, Csize_t, Ptr{Int64}, Ptr{Int64}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6)
end

function cholmod_csymamd(A, Cmember, Perm, Common)
    ccall((:cholmod_csymamd, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{Int32}, Ptr{Int32}, Ptr{cholmod_common}), A, Cmember, Perm, Common)
end

function cholmod_l_csymamd(arg1, arg2, arg3, arg4)
    ccall((:cholmod_l_csymamd, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{Int64}, Ptr{Int64}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4)
end

function cholmod_camd(A, fset, fsize, Cmember, Perm, Common)
    ccall((:cholmod_camd, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{Int32}, Csize_t, Ptr{Int32}, Ptr{Int32}, Ptr{cholmod_common}), A, fset, fsize, Cmember, Perm, Common)
end

function cholmod_l_camd(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:cholmod_l_camd, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{Int64}, Csize_t, Ptr{Int64}, Ptr{Int64}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6)
end

function cholmod_nested_dissection(A, fset, fsize, Perm, CParent, Cmember, Common)
    ccall((:cholmod_nested_dissection, libcholmod), Int64, (Ptr{cholmod_sparse}, Ptr{Int32}, Csize_t, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{cholmod_common}), A, fset, fsize, Perm, CParent, Cmember, Common)
end

function cholmod_l_nested_dissection(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    ccall((:cholmod_l_nested_dissection, libcholmod), Int64, (Ptr{cholmod_sparse}, Ptr{Int64}, Csize_t, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6, arg7)
end

function cholmod_metis(A, fset, fsize, postorder, Perm, Common)
    ccall((:cholmod_metis, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{Int32}, Csize_t, Cint, Ptr{Int32}, Ptr{cholmod_common}), A, fset, fsize, postorder, Perm, Common)
end

function cholmod_l_metis(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:cholmod_l_metis, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{Int64}, Csize_t, Cint, Ptr{Int64}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6)
end

function cholmod_bisect(A, fset, fsize, compress, Partition, Common)
    ccall((:cholmod_bisect, libcholmod), Int64, (Ptr{cholmod_sparse}, Ptr{Int32}, Csize_t, Cint, Ptr{Int32}, Ptr{cholmod_common}), A, fset, fsize, compress, Partition, Common)
end

function cholmod_l_bisect(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:cholmod_l_bisect, libcholmod), Int64, (Ptr{cholmod_sparse}, Ptr{Int64}, Csize_t, Cint, Ptr{Int64}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6)
end

function cholmod_metis_bisector(A, Anw, Aew, Partition, Common)
    ccall((:cholmod_metis_bisector, libcholmod), Int64, (Ptr{cholmod_sparse}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{cholmod_common}), A, Anw, Aew, Partition, Common)
end

function cholmod_l_metis_bisector(arg1, arg2, arg3, arg4, arg5)
    ccall((:cholmod_l_metis_bisector, libcholmod), Int64, (Ptr{cholmod_sparse}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5)
end

function cholmod_collapse_septree(n, ncomponents, nd_oksep, nd_small, CParent, Cmember, Common)
    ccall((:cholmod_collapse_septree, libcholmod), Int64, (Csize_t, Csize_t, Cdouble, Csize_t, Ptr{Int32}, Ptr{Int32}, Ptr{cholmod_common}), n, ncomponents, nd_oksep, nd_small, CParent, Cmember, Common)
end

function cholmod_l_collapse_septree(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    ccall((:cholmod_l_collapse_septree, libcholmod), Int64, (Csize_t, Csize_t, Cdouble, Csize_t, Ptr{Int64}, Ptr{Int64}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6, arg7)
end

function cholmod_super_symbolic(A, F, Parent, L, Common)
    ccall((:cholmod_super_symbolic, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{cholmod_sparse}, Ptr{Int32}, Ptr{cholmod_factor}, Ptr{cholmod_common}), A, F, Parent, L, Common)
end

function cholmod_l_super_symbolic(arg1, arg2, arg3, arg4, arg5)
    ccall((:cholmod_l_super_symbolic, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{cholmod_sparse}, Ptr{Int64}, Ptr{cholmod_factor}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5)
end

function cholmod_super_symbolic2(for_whom, A, F, Parent, L, Common)
    ccall((:cholmod_super_symbolic2, libcholmod), Cint, (Cint, Ptr{cholmod_sparse}, Ptr{cholmod_sparse}, Ptr{Int32}, Ptr{cholmod_factor}, Ptr{cholmod_common}), for_whom, A, F, Parent, L, Common)
end

function cholmod_l_super_symbolic2(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:cholmod_l_super_symbolic2, libcholmod), Cint, (Cint, Ptr{cholmod_sparse}, Ptr{cholmod_sparse}, Ptr{Int64}, Ptr{cholmod_factor}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5, arg6)
end

function cholmod_super_numeric(A, F, beta, L, Common)
    ccall((:cholmod_super_numeric, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{cholmod_sparse}, Ptr{Cdouble}, Ptr{cholmod_factor}, Ptr{cholmod_common}), A, F, beta, L, Common)
end

function cholmod_l_super_numeric(arg1, arg2, arg3, arg4, arg5)
    ccall((:cholmod_l_super_numeric, libcholmod), Cint, (Ptr{cholmod_sparse}, Ptr{cholmod_sparse}, Ptr{Cdouble}, Ptr{cholmod_factor}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4, arg5)
end

function cholmod_super_lsolve(L, X, E, Common)
    ccall((:cholmod_super_lsolve, libcholmod), Cint, (Ptr{cholmod_factor}, Ptr{cholmod_dense}, Ptr{cholmod_dense}, Ptr{cholmod_common}), L, X, E, Common)
end

function cholmod_l_super_lsolve(arg1, arg2, arg3, arg4)
    ccall((:cholmod_l_super_lsolve, libcholmod), Cint, (Ptr{cholmod_factor}, Ptr{cholmod_dense}, Ptr{cholmod_dense}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4)
end

function cholmod_super_ltsolve(L, X, E, Common)
    ccall((:cholmod_super_ltsolve, libcholmod), Cint, (Ptr{cholmod_factor}, Ptr{cholmod_dense}, Ptr{cholmod_dense}, Ptr{cholmod_common}), L, X, E, Common)
end

function cholmod_l_super_ltsolve(arg1, arg2, arg3, arg4)
    ccall((:cholmod_l_super_ltsolve, libcholmod), Cint, (Ptr{cholmod_factor}, Ptr{cholmod_dense}, Ptr{cholmod_dense}, Ptr{cholmod_common}), arg1, arg2, arg3, arg4)
end

function SuiteSparseQR_C(ordering, tol, econ, getCTX, A, Bsparse, Bdense, Zsparse, Zdense, R, E, H, HPinv, HTau, cc)
    ccall((:SuiteSparseQR_C, libspqr), Int64, (Cint, Cdouble, Int64, Cint, Ptr{cholmod_sparse}, Ptr{cholmod_sparse}, Ptr{cholmod_dense}, Ptr{Ptr{cholmod_sparse}}, Ptr{Ptr{cholmod_dense}}, Ptr{Ptr{cholmod_sparse}}, Ptr{Ptr{Int64}}, Ptr{Ptr{cholmod_sparse}}, Ptr{Ptr{Int64}}, Ptr{Ptr{cholmod_dense}}, Ptr{cholmod_common}), ordering, tol, econ, getCTX, A, Bsparse, Bdense, Zsparse, Zdense, R, E, H, HPinv, HTau, cc)
end

function SuiteSparseQR_C_QR(ordering, tol, econ, A, Q, R, E, cc)
    ccall((:SuiteSparseQR_C_QR, libspqr), Int64, (Cint, Cdouble, Int64, Ptr{cholmod_sparse}, Ptr{Ptr{cholmod_sparse}}, Ptr{Ptr{cholmod_sparse}}, Ptr{Ptr{Int64}}, Ptr{cholmod_common}), ordering, tol, econ, A, Q, R, E, cc)
end

function SuiteSparseQR_C_backslash(ordering, tol, A, B, cc)
    ccall((:SuiteSparseQR_C_backslash, libspqr), Ptr{cholmod_dense}, (Cint, Cdouble, Ptr{cholmod_sparse}, Ptr{cholmod_dense}, Ptr{cholmod_common}), ordering, tol, A, B, cc)
end

function SuiteSparseQR_C_backslash_default(A, B, cc)
    ccall((:SuiteSparseQR_C_backslash_default, libspqr), Ptr{cholmod_dense}, (Ptr{cholmod_sparse}, Ptr{cholmod_dense}, Ptr{cholmod_common}), A, B, cc)
end

function SuiteSparseQR_C_backslash_sparse(ordering, tol, A, B, cc)
    ccall((:SuiteSparseQR_C_backslash_sparse, libspqr), Ptr{cholmod_sparse}, (Cint, Cdouble, Ptr{cholmod_sparse}, Ptr{cholmod_sparse}, Ptr{cholmod_common}), ordering, tol, A, B, cc)
end

mutable struct SuiteSparseQR_C_factorization_struct
    xtype::Cint
    factors::Ptr{Cvoid}
    SuiteSparseQR_C_factorization_struct() = new()
end

const SuiteSparseQR_C_factorization = SuiteSparseQR_C_factorization_struct

function SuiteSparseQR_C_factorize(ordering, tol, A, cc)
    ccall((:SuiteSparseQR_C_factorize, libspqr), Ptr{SuiteSparseQR_C_factorization}, (Cint, Cdouble, Ptr{cholmod_sparse}, Ptr{cholmod_common}), ordering, tol, A, cc)
end

function SuiteSparseQR_C_symbolic(ordering, allow_tol, A, cc)
    ccall((:SuiteSparseQR_C_symbolic, libspqr), Ptr{SuiteSparseQR_C_factorization}, (Cint, Cint, Ptr{cholmod_sparse}, Ptr{cholmod_common}), ordering, allow_tol, A, cc)
end

function SuiteSparseQR_C_numeric(tol, A, QR, cc)
    ccall((:SuiteSparseQR_C_numeric, libspqr), Cint, (Cdouble, Ptr{cholmod_sparse}, Ptr{SuiteSparseQR_C_factorization}, Ptr{cholmod_common}), tol, A, QR, cc)
end

function SuiteSparseQR_C_free(QR, cc)
    ccall((:SuiteSparseQR_C_free, libspqr), Cint, (Ptr{Ptr{SuiteSparseQR_C_factorization}}, Ptr{cholmod_common}), QR, cc)
end

function SuiteSparseQR_C_solve(system, QR, B, cc)
    ccall((:SuiteSparseQR_C_solve, libspqr), Ptr{cholmod_dense}, (Cint, Ptr{SuiteSparseQR_C_factorization}, Ptr{cholmod_dense}, Ptr{cholmod_common}), system, QR, B, cc)
end

function SuiteSparseQR_C_qmult(method, QR, X, cc)
    ccall((:SuiteSparseQR_C_qmult, libspqr), Ptr{cholmod_dense}, (Cint, Ptr{SuiteSparseQR_C_factorization}, Ptr{cholmod_dense}, Ptr{cholmod_common}), method, QR, X, cc)
end

function amd_order(n, Ap, Ai, P, Control, Info)
    ccall((:amd_order, libamd), Cint, (Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble}), n, Ap, Ai, P, Control, Info)
end

function amd_l_order(n, Ap, Ai, P, Control, Info)
    ccall((:amd_l_order, libamd), Cint, (Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cdouble}), n, Ap, Ai, P, Control, Info)
end

function amd_2(n, Pe, Iw, Len, iwlen, pfree, Nv, Next, Last, Head, Elen, Degree, W, Control, Info)
    ccall((:amd_2, libamd), Cvoid, (Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Int32, Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble}), n, Pe, Iw, Len, iwlen, pfree, Nv, Next, Last, Head, Elen, Degree, W, Control, Info)
end

function amd_l2(n, Pe, Iw, Len, iwlen, pfree, Nv, Next, Last, Head, Elen, Degree, W, Control, Info)
    ccall((:amd_l2, libamd), Cvoid, (Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Int64, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cdouble}), n, Pe, Iw, Len, iwlen, pfree, Nv, Next, Last, Head, Elen, Degree, W, Control, Info)
end

function amd_valid(n_row, n_col, Ap, Ai)
    ccall((:amd_valid, libamd), Cint, (Int32, Int32, Ptr{Int32}, Ptr{Int32}), n_row, n_col, Ap, Ai)
end

function amd_l_valid(n_row, n_col, Ap, Ai)
    ccall((:amd_l_valid, libamd), Cint, (Int64, Int64, Ptr{Int64}, Ptr{Int64}), n_row, n_col, Ap, Ai)
end

function amd_defaults(Control)
    ccall((:amd_defaults, libamd), Cvoid, (Ptr{Cdouble},), Control)
end

function amd_l_defaults(Control)
    ccall((:amd_l_defaults, libamd), Cvoid, (Ptr{Cdouble},), Control)
end

function amd_control(Control)
    ccall((:amd_control, libamd), Cvoid, (Ptr{Cdouble},), Control)
end

function amd_l_control(Control)
    ccall((:amd_l_control, libamd), Cvoid, (Ptr{Cdouble},), Control)
end

function amd_info(Info)
    ccall((:amd_info, libamd), Cvoid, (Ptr{Cdouble},), Info)
end

function amd_l_info(Info)
    ccall((:amd_l_info, libamd), Cvoid, (Ptr{Cdouble},), Info)
end

function colamd_recommended(nnz, n_row, n_col)
    ccall((:colamd_recommended, libcolamd), Csize_t, (Int32, Int32, Int32), nnz, n_row, n_col)
end

function colamd_l_recommended(nnz, n_row, n_col)
    ccall((:colamd_l_recommended, libcolamd), Csize_t, (Int64, Int64, Int64), nnz, n_row, n_col)
end

function colamd_set_defaults(knobs)
    ccall((:colamd_set_defaults, libcolamd), Cvoid, (Ptr{Cdouble},), knobs)
end

function colamd_l_set_defaults(knobs)
    ccall((:colamd_l_set_defaults, libcolamd), Cvoid, (Ptr{Cdouble},), knobs)
end

function colamd(n_row, n_col, Alen, A, p, knobs, stats)
    ccall((:colamd, libcolamd), Cint, (Int32, Int32, Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Int32}), n_row, n_col, Alen, A, p, knobs, stats)
end

function colamd_l(n_row, n_col, Alen, A, p, knobs, stats)
    ccall((:colamd_l, libcolamd), Cint, (Int64, Int64, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Int64}), n_row, n_col, Alen, A, p, knobs, stats)
end

function symamd(n, A, p, perm, knobs, stats, allocate, release)
    ccall((:symamd, libcolamd), Cint, (Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Int32}, Ptr{Cvoid}, Ptr{Cvoid}), n, A, p, perm, knobs, stats, allocate, release)
end

function symamd_l(n, A, p, perm, knobs, stats, allocate, release)
    ccall((:symamd_l, libcolamd), Cint, (Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Int64}, Ptr{Cvoid}, Ptr{Cvoid}), n, A, p, perm, knobs, stats, allocate, release)
end

function colamd_report(stats)
    ccall((:colamd_report, libcolamd), Cvoid, (Ptr{Int32},), stats)
end

function colamd_l_report(stats)
    ccall((:colamd_l_report, libcolamd), Cvoid, (Ptr{Int64},), stats)
end

function symamd_report(stats)
    ccall((:symamd_report, libcolamd), Cvoid, (Ptr{Int32},), stats)
end

function symamd_l_report(stats)
    ccall((:symamd_l_report, libcolamd), Cvoid, (Ptr{Int64},), stats)
end

function ccolamd_recommended(nnz, n_row, n_col)
    ccall((:ccolamd_recommended, libcolamd), Csize_t, (Cint, Cint, Cint), nnz, n_row, n_col)
end

function ccolamd_l_recommended(nnz, n_row, n_col)
    ccall((:ccolamd_l_recommended, libcolamd), Csize_t, (Int64, Int64, Int64), nnz, n_row, n_col)
end

function ccolamd_set_defaults(knobs)
    ccall((:ccolamd_set_defaults, libcolamd), Cvoid, (Ptr{Cdouble},), knobs)
end

function ccolamd_l_set_defaults(knobs)
    ccall((:ccolamd_l_set_defaults, libcolamd), Cvoid, (Ptr{Cdouble},), knobs)
end

function ccolamd(n_row, n_col, Alen, A, p, knobs, stats, cmember)
    ccall((:ccolamd, libcolamd), Cint, (Cint, Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}), n_row, n_col, Alen, A, p, knobs, stats, cmember)
end

function ccolamd_l(n_row, n_col, Alen, A, p, knobs, stats, cmember)
    ccall((:ccolamd_l, libcolamd), Cint, (Int64, Int64, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Int64}, Ptr{Int64}), n_row, n_col, Alen, A, p, knobs, stats, cmember)
end

function csymamd(n, A, p, perm, knobs, stats, allocate, release, cmember, stype)
    ccall((:csymamd, libcolamd), Cint, (Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cint}, Cint), n, A, p, perm, knobs, stats, allocate, release, cmember, stype)
end

function csymamd_l(n, A, p, perm, knobs, stats, allocate, release, cmember, stype)
    ccall((:csymamd_l, libcolamd), Cint, (Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Int64}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Int64}, Int64), n, A, p, perm, knobs, stats, allocate, release, cmember, stype)
end

function ccolamd_report(stats)
    ccall((:ccolamd_report, libcolamd), Cvoid, (Ptr{Cint},), stats)
end

function ccolamd_l_report(stats)
    ccall((:ccolamd_l_report, libcolamd), Cvoid, (Ptr{Int64},), stats)
end

function csymamd_report(stats)
    ccall((:csymamd_report, libcolamd), Cvoid, (Ptr{Cint},), stats)
end

function csymamd_l_report(stats)
    ccall((:csymamd_l_report, libcolamd), Cvoid, (Ptr{Int64},), stats)
end

function ccolamd2(n_row, n_col, Alen, A, p, knobs, stats, Front_npivcol, Front_nrows, Front_ncols, Front_parent, Front_cols, p_nfr, InFront, cmember)
    ccall((:ccolamd2, libcolamd), Cint, (Cint, Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), n_row, n_col, Alen, A, p, knobs, stats, Front_npivcol, Front_nrows, Front_ncols, Front_parent, Front_cols, p_nfr, InFront, cmember)
end

function ccolamd2_l(n_row, n_col, Alen, A, p, knobs, stats, Front_npivcol, Front_nrows, Front_ncols, Front_parent, Front_cols, p_nfr, InFront, cmember)
    ccall((:ccolamd2_l, libcolamd), Cint, (Int64, Int64, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}), n_row, n_col, Alen, A, p, knobs, stats, Front_npivcol, Front_nrows, Front_ncols, Front_parent, Front_cols, p_nfr, InFront, cmember)
end

function ccolamd_apply_order(Front, Order, Temp, nn, nfr)
    ccall((:ccolamd_apply_order, libcolamd), Cvoid, (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Cint, Cint), Front, Order, Temp, nn, nfr)
end

function ccolamd_l_apply_order(Front, Order, Temp, nn, nfr)
    ccall((:ccolamd_l_apply_order, libcolamd), Cvoid, (Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Int64, Int64), Front, Order, Temp, nn, nfr)
end

function ccolamd_fsize(nn, MaxFsize, Fnrows, Fncols, Parent, Npiv)
    ccall((:ccolamd_fsize, libcolamd), Cvoid, (Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), nn, MaxFsize, Fnrows, Fncols, Parent, Npiv)
end

function ccolamd_l_fsize(nn, MaxFsize, Fnrows, Fncols, Parent, Npiv)
    ccall((:ccolamd_l_fsize, libcolamd), Cvoid, (Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}), nn, MaxFsize, Fnrows, Fncols, Parent, Npiv)
end

function ccolamd_postorder(nn, Parent, Npiv, Fsize, Order, Child, Sibling, Stack, Front_cols, cmember)
    ccall((:ccolamd_postorder, libcolamd), Cvoid, (Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), nn, Parent, Npiv, Fsize, Order, Child, Sibling, Stack, Front_cols, cmember)
end

function ccolamd_l_postorder(nn, Parent, Npiv, Fsize, Order, Child, Sibling, Stack, Front_cols, cmember)
    ccall((:ccolamd_l_postorder, libcolamd), Cvoid, (Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}), nn, Parent, Npiv, Fsize, Order, Child, Sibling, Stack, Front_cols, cmember)
end

function ccolamd_post_tree(root, k, Child, Sibling, Order, Stack)
    ccall((:ccolamd_post_tree, libcolamd), Cint, (Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), root, k, Child, Sibling, Order, Stack)
end

function ccolamd_l_post_tree(root, k, Child, Sibling, Order, Stack)
    ccall((:ccolamd_l_post_tree, libcolamd), Int64, (Int64, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}), root, k, Child, Sibling, Order, Stack)
end

function umfpack_di_symbolic(n_row, n_col, Ap, Ai, Ax, Symbolic, Control, Info)
    ccall((:umfpack_di_symbolic, libumfpack), Cint, (Int32, Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Ptr{Cvoid}}, Ptr{Cdouble}, Ptr{Cdouble}), n_row, n_col, Ap, Ai, Ax, Symbolic, Control, Info)
end

function umfpack_dl_symbolic(n_row, n_col, Ap, Ai, Ax, Symbolic, Control, Info)
    ccall((:umfpack_dl_symbolic, libumfpack), Cint, (Int64, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Ptr{Cvoid}}, Ptr{Cdouble}, Ptr{Cdouble}), n_row, n_col, Ap, Ai, Ax, Symbolic, Control, Info)
end

function umfpack_zi_symbolic(n_row, n_col, Ap, Ai, Ax, Az, Symbolic, Control, Info)
    ccall((:umfpack_zi_symbolic, libumfpack), Cint, (Int32, Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Ptr{Cvoid}}, Ptr{Cdouble}, Ptr{Cdouble}), n_row, n_col, Ap, Ai, Ax, Az, Symbolic, Control, Info)
end

function umfpack_zl_symbolic(n_row, n_col, Ap, Ai, Ax, Az, Symbolic, Control, Info)
    ccall((:umfpack_zl_symbolic, libumfpack), Cint, (Int64, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Ptr{Cvoid}}, Ptr{Cdouble}, Ptr{Cdouble}), n_row, n_col, Ap, Ai, Ax, Az, Symbolic, Control, Info)
end

function umfpack_di_numeric(Ap, Ai, Ax, Symbolic, Numeric, Control, Info)
    ccall((:umfpack_di_numeric, libumfpack), Cint, (Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Ptr{Cdouble}, Ptr{Cdouble}), Ap, Ai, Ax, Symbolic, Numeric, Control, Info)
end

function umfpack_dl_numeric(Ap, Ai, Ax, Symbolic, Numeric, Control, Info)
    ccall((:umfpack_dl_numeric, libumfpack), Cint, (Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Ptr{Cdouble}, Ptr{Cdouble}), Ap, Ai, Ax, Symbolic, Numeric, Control, Info)
end

function umfpack_zi_numeric(Ap, Ai, Ax, Az, Symbolic, Numeric, Control, Info)
    ccall((:umfpack_zi_numeric, libumfpack), Cint, (Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Ptr{Cdouble}, Ptr{Cdouble}), Ap, Ai, Ax, Az, Symbolic, Numeric, Control, Info)
end

function umfpack_zl_numeric(Ap, Ai, Ax, Az, Symbolic, Numeric, Control, Info)
    ccall((:umfpack_zl_numeric, libumfpack), Cint, (Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Ptr{Cdouble}, Ptr{Cdouble}), Ap, Ai, Ax, Az, Symbolic, Numeric, Control, Info)
end

function umfpack_di_solve(sys, Ap, Ai, Ax, X, B, Numeric, Control, Info)
    ccall((:umfpack_di_solve, libumfpack), Cint, (Cint, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}), sys, Ap, Ai, Ax, X, B, Numeric, Control, Info)
end

function umfpack_dl_solve(sys, Ap, Ai, Ax, X, B, Numeric, Control, Info)
    ccall((:umfpack_dl_solve, libumfpack), Cint, (Cint, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}), sys, Ap, Ai, Ax, X, B, Numeric, Control, Info)
end

function umfpack_zi_solve(sys, Ap, Ai, Ax, Az, Xx, Xz, Bx, Bz, Numeric, Control, Info)
    ccall((:umfpack_zi_solve, libumfpack), Cint, (Cint, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}), sys, Ap, Ai, Ax, Az, Xx, Xz, Bx, Bz, Numeric, Control, Info)
end

function umfpack_zl_solve(sys, Ap, Ai, Ax, Az, Xx, Xz, Bx, Bz, Numeric, Control, Info)
    ccall((:umfpack_zl_solve, libumfpack), Cint, (Cint, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}), sys, Ap, Ai, Ax, Az, Xx, Xz, Bx, Bz, Numeric, Control, Info)
end

function umfpack_di_free_symbolic(Symbolic)
    ccall((:umfpack_di_free_symbolic, libumfpack), Cvoid, (Ptr{Ptr{Cvoid}},), Symbolic)
end

function umfpack_dl_free_symbolic(Symbolic)
    ccall((:umfpack_dl_free_symbolic, libumfpack), Cvoid, (Ptr{Ptr{Cvoid}},), Symbolic)
end

function umfpack_zi_free_symbolic(Symbolic)
    ccall((:umfpack_zi_free_symbolic, libumfpack), Cvoid, (Ptr{Ptr{Cvoid}},), Symbolic)
end

function umfpack_zl_free_symbolic(Symbolic)
    ccall((:umfpack_zl_free_symbolic, libumfpack), Cvoid, (Ptr{Ptr{Cvoid}},), Symbolic)
end

function umfpack_di_free_numeric(Numeric)
    ccall((:umfpack_di_free_numeric, libumfpack), Cvoid, (Ptr{Ptr{Cvoid}},), Numeric)
end

function umfpack_dl_free_numeric(Numeric)
    ccall((:umfpack_dl_free_numeric, libumfpack), Cvoid, (Ptr{Ptr{Cvoid}},), Numeric)
end

function umfpack_zi_free_numeric(Numeric)
    ccall((:umfpack_zi_free_numeric, libumfpack), Cvoid, (Ptr{Ptr{Cvoid}},), Numeric)
end

function umfpack_zl_free_numeric(Numeric)
    ccall((:umfpack_zl_free_numeric, libumfpack), Cvoid, (Ptr{Ptr{Cvoid}},), Numeric)
end

function umfpack_di_defaults(Control)
    ccall((:umfpack_di_defaults, libumfpack), Cvoid, (Ptr{Cdouble},), Control)
end

function umfpack_dl_defaults(Control)
    ccall((:umfpack_dl_defaults, libumfpack), Cvoid, (Ptr{Cdouble},), Control)
end

function umfpack_zi_defaults(Control)
    ccall((:umfpack_zi_defaults, libumfpack), Cvoid, (Ptr{Cdouble},), Control)
end

function umfpack_zl_defaults(Control)
    ccall((:umfpack_zl_defaults, libumfpack), Cvoid, (Ptr{Cdouble},), Control)
end

function umfpack_di_qsymbolic(n_row, n_col, Ap, Ai, Ax, Qinit, Symbolic, Control, Info)
    ccall((:umfpack_di_qsymbolic, libumfpack), Cint, (Int32, Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Int32}, Ptr{Ptr{Cvoid}}, Ptr{Cdouble}, Ptr{Cdouble}), n_row, n_col, Ap, Ai, Ax, Qinit, Symbolic, Control, Info)
end

function umfpack_dl_qsymbolic(n_row, n_col, Ap, Ai, Ax, Qinit, Symbolic, Control, Info)
    ccall((:umfpack_dl_qsymbolic, libumfpack), Cint, (Int64, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Int64}, Ptr{Ptr{Cvoid}}, Ptr{Cdouble}, Ptr{Cdouble}), n_row, n_col, Ap, Ai, Ax, Qinit, Symbolic, Control, Info)
end

function umfpack_zi_qsymbolic(n_row, n_col, Ap, Ai, Ax, Az, Qinit, Symbolic, Control, Info)
    ccall((:umfpack_zi_qsymbolic, libumfpack), Cint, (Int32, Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Int32}, Ptr{Ptr{Cvoid}}, Ptr{Cdouble}, Ptr{Cdouble}), n_row, n_col, Ap, Ai, Ax, Az, Qinit, Symbolic, Control, Info)
end

function umfpack_zl_qsymbolic(n_row, n_col, Ap, Ai, Ax, Az, Qinit, Symbolic, Control, Info)
    ccall((:umfpack_zl_qsymbolic, libumfpack), Cint, (Int64, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Int64}, Ptr{Ptr{Cvoid}}, Ptr{Cdouble}, Ptr{Cdouble}), n_row, n_col, Ap, Ai, Ax, Az, Qinit, Symbolic, Control, Info)
end

function umfpack_di_fsymbolic(n_row, n_col, Ap, Ai, Ax, user_ordering, user_params, Symbolic, Control, Info)
    ccall((:umfpack_di_fsymbolic, libumfpack), Cint, (Int32, Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Ptr{Cdouble}, Ptr{Cdouble}), n_row, n_col, Ap, Ai, Ax, user_ordering, user_params, Symbolic, Control, Info)
end

function umfpack_dl_fsymbolic(n_row, n_col, Ap, Ai, Ax, user_ordering, user_params, Symbolic, Control, Info)
    ccall((:umfpack_dl_fsymbolic, libumfpack), Cint, (Int64, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Ptr{Cdouble}, Ptr{Cdouble}), n_row, n_col, Ap, Ai, Ax, user_ordering, user_params, Symbolic, Control, Info)
end

function umfpack_zi_fsymbolic(n_row, n_col, Ap, Ai, Ax, Az, user_ordering, user_params, Symbolic, Control, Info)
    ccall((:umfpack_zi_fsymbolic, libumfpack), Cint, (Int32, Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Ptr{Cdouble}, Ptr{Cdouble}), n_row, n_col, Ap, Ai, Ax, Az, user_ordering, user_params, Symbolic, Control, Info)
end

function umfpack_zl_fsymbolic(n_row, n_col, Ap, Ai, Ax, Az, user_ordering, user_params, Symbolic, Control, Info)
    ccall((:umfpack_zl_fsymbolic, libumfpack), Cint, (Int64, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Ptr{Cdouble}, Ptr{Cdouble}), n_row, n_col, Ap, Ai, Ax, Az, user_ordering, user_params, Symbolic, Control, Info)
end

function umfpack_di_paru_symbolic(n_row, n_col, Ap, Ai, Ax, Qinit, user_ordering, user_params, Symbolic, SW, Control, Info)
    ccall((:umfpack_di_paru_symbolic, libumfpack), Cint, (Int32, Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Int32}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, Ptr{Cdouble}, Ptr{Cdouble}), n_row, n_col, Ap, Ai, Ax, Qinit, user_ordering, user_params, Symbolic, SW, Control, Info)
end

function umfpack_dl_paru_symbolic(n_row, n_col, Ap, Ai, Ax, Qinit, user_ordering, user_params, Symbolic, SW, Control, Info)
    ccall((:umfpack_dl_paru_symbolic, libumfpack), Cint, (Int64, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Int64}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, Ptr{Cdouble}, Ptr{Cdouble}), n_row, n_col, Ap, Ai, Ax, Qinit, user_ordering, user_params, Symbolic, SW, Control, Info)
end

function umfpack_zi_paru_symbolic(n_row, n_col, Ap, Ai, Ax, Az, Qinit, user_ordering, user_params, Symbolic, SW, Control, Info)
    ccall((:umfpack_zi_paru_symbolic, libumfpack), Cint, (Int32, Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Int32}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, Ptr{Cdouble}, Ptr{Cdouble}), n_row, n_col, Ap, Ai, Ax, Az, Qinit, user_ordering, user_params, Symbolic, SW, Control, Info)
end

function umfpack_zl_paru_symbolic(n_row, n_col, Ap, Ai, Ax, Az, Qinit, user_ordering, user_params, Symbolic, SW, Control, Info)
    ccall((:umfpack_zl_paru_symbolic, libumfpack), Cint, (Int64, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Int64}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, Ptr{Cdouble}, Ptr{Cdouble}), n_row, n_col, Ap, Ai, Ax, Az, Qinit, user_ordering, user_params, Symbolic, SW, Control, Info)
end

function umfpack_di_paru_free_sw(SW)
    ccall((:umfpack_di_paru_free_sw, libumfpack), Cvoid, (Ptr{Ptr{Cvoid}},), SW)
end

function umfpack_dl_paru_free_sw(SW)
    ccall((:umfpack_dl_paru_free_sw, libumfpack), Cvoid, (Ptr{Ptr{Cvoid}},), SW)
end

function umfpack_zi_paru_free_sw(SW)
    ccall((:umfpack_zi_paru_free_sw, libumfpack), Cvoid, (Ptr{Ptr{Cvoid}},), SW)
end

function umfpack_zl_paru_free_sw(SW)
    ccall((:umfpack_zl_paru_free_sw, libumfpack), Cvoid, (Ptr{Ptr{Cvoid}},), SW)
end

function umfpack_di_wsolve(sys, Ap, Ai, Ax, X, B, Numeric, Control, Info, Wi, W)
    ccall((:umfpack_di_wsolve, libumfpack), Cint, (Cint, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Int32}, Ptr{Cdouble}), sys, Ap, Ai, Ax, X, B, Numeric, Control, Info, Wi, W)
end

function umfpack_dl_wsolve(sys, Ap, Ai, Ax, X, B, Numeric, Control, Info, Wi, W)
    ccall((:umfpack_dl_wsolve, libumfpack), Cint, (Cint, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Int64}, Ptr{Cdouble}), sys, Ap, Ai, Ax, X, B, Numeric, Control, Info, Wi, W)
end

function umfpack_zi_wsolve(sys, Ap, Ai, Ax, Az, Xx, Xz, Bx, Bz, Numeric, Control, Info, Wi, W)
    ccall((:umfpack_zi_wsolve, libumfpack), Cint, (Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Int32}, Ptr{Cdouble}), sys, Ap, Ai, Ax, Az, Xx, Xz, Bx, Bz, Numeric, Control, Info, Wi, W)
end

function umfpack_zl_wsolve(sys, Ap, Ai, Ax, Az, Xx, Xz, Bx, Bz, Numeric, Control, Info, Wi, W)
    ccall((:umfpack_zl_wsolve, libumfpack), Cint, (Cint, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Int64}, Ptr{Cdouble}), sys, Ap, Ai, Ax, Az, Xx, Xz, Bx, Bz, Numeric, Control, Info, Wi, W)
end

function umfpack_di_triplet_to_col(n_row, n_col, nz, Ti, Tj, Tx, Ap, Ai, Ax, Map)
    ccall((:umfpack_di_triplet_to_col, libumfpack), Cint, (Int32, Int32, Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Int32}), n_row, n_col, nz, Ti, Tj, Tx, Ap, Ai, Ax, Map)
end

function umfpack_dl_triplet_to_col(n_row, n_col, nz, Ti, Tj, Tx, Ap, Ai, Ax, Map)
    ccall((:umfpack_dl_triplet_to_col, libumfpack), Cint, (Int64, Int64, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Int64}), n_row, n_col, nz, Ti, Tj, Tx, Ap, Ai, Ax, Map)
end

function umfpack_zi_triplet_to_col(n_row, n_col, nz, Ti, Tj, Tx, Tz, Ap, Ai, Ax, Az, Map)
    ccall((:umfpack_zi_triplet_to_col, libumfpack), Cint, (Int32, Int32, Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Int32}), n_row, n_col, nz, Ti, Tj, Tx, Tz, Ap, Ai, Ax, Az, Map)
end

function umfpack_zl_triplet_to_col(n_row, n_col, nz, Ti, Tj, Tx, Tz, Ap, Ai, Ax, Az, Map)
    ccall((:umfpack_zl_triplet_to_col, libumfpack), Cint, (Int64, Int64, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Int64}), n_row, n_col, nz, Ti, Tj, Tx, Tz, Ap, Ai, Ax, Az, Map)
end

function umfpack_di_col_to_triplet(n_col, Ap, Tj)
    ccall((:umfpack_di_col_to_triplet, libumfpack), Cint, (Int32, Ptr{Int32}, Ptr{Int32}), n_col, Ap, Tj)
end

function umfpack_dl_col_to_triplet(n_col, Ap, Tj)
    ccall((:umfpack_dl_col_to_triplet, libumfpack), Cint, (Int64, Ptr{Int64}, Ptr{Int64}), n_col, Ap, Tj)
end

function umfpack_zi_col_to_triplet(n_col, Ap, Tj)
    ccall((:umfpack_zi_col_to_triplet, libumfpack), Cint, (Int32, Ptr{Int32}, Ptr{Int32}), n_col, Ap, Tj)
end

function umfpack_zl_col_to_triplet(n_col, Ap, Tj)
    ccall((:umfpack_zl_col_to_triplet, libumfpack), Cint, (Int64, Ptr{Int64}, Ptr{Int64}), n_col, Ap, Tj)
end

function umfpack_di_transpose(n_row, n_col, Ap, Ai, Ax, P, Q, Rp, Ri, Rx)
    ccall((:umfpack_di_transpose, libumfpack), Cint, (Int32, Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}), n_row, n_col, Ap, Ai, Ax, P, Q, Rp, Ri, Rx)
end

function umfpack_dl_transpose(n_row, n_col, Ap, Ai, Ax, P, Q, Rp, Ri, Rx)
    ccall((:umfpack_dl_transpose, libumfpack), Cint, (Int64, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}), n_row, n_col, Ap, Ai, Ax, P, Q, Rp, Ri, Rx)
end

function umfpack_zi_transpose(n_row, n_col, Ap, Ai, Ax, Az, P, Q, Rp, Ri, Rx, Rz, do_conjugate)
    ccall((:umfpack_zi_transpose, libumfpack), Cint, (Int32, Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble}, Cint), n_row, n_col, Ap, Ai, Ax, Az, P, Q, Rp, Ri, Rx, Rz, do_conjugate)
end

function umfpack_zl_transpose(n_row, n_col, Ap, Ai, Ax, Az, P, Q, Rp, Ri, Rx, Rz, do_conjugate)
    ccall((:umfpack_zl_transpose, libumfpack), Cint, (Int64, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cdouble}, Cint), n_row, n_col, Ap, Ai, Ax, Az, P, Q, Rp, Ri, Rx, Rz, do_conjugate)
end

function umfpack_di_scale(X, B, Numeric)
    ccall((:umfpack_di_scale, libumfpack), Cint, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}), X, B, Numeric)
end

function umfpack_dl_scale(X, B, Numeric)
    ccall((:umfpack_dl_scale, libumfpack), Cint, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}), X, B, Numeric)
end

function umfpack_zi_scale(Xx, Xz, Bx, Bz, Numeric)
    ccall((:umfpack_zi_scale, libumfpack), Cint, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}), Xx, Xz, Bx, Bz, Numeric)
end

function umfpack_zl_scale(Xx, Xz, Bx, Bz, Numeric)
    ccall((:umfpack_zl_scale, libumfpack), Cint, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}), Xx, Xz, Bx, Bz, Numeric)
end

function umfpack_di_get_lunz(lnz, unz, n_row, n_col, nz_udiag, Numeric)
    ccall((:umfpack_di_get_lunz, libumfpack), Cint, (Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Cvoid}), lnz, unz, n_row, n_col, nz_udiag, Numeric)
end

function umfpack_dl_get_lunz(lnz, unz, n_row, n_col, nz_udiag, Numeric)
    ccall((:umfpack_dl_get_lunz, libumfpack), Cint, (Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Cvoid}), lnz, unz, n_row, n_col, nz_udiag, Numeric)
end

function umfpack_zi_get_lunz(lnz, unz, n_row, n_col, nz_udiag, Numeric)
    ccall((:umfpack_zi_get_lunz, libumfpack), Cint, (Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Cvoid}), lnz, unz, n_row, n_col, nz_udiag, Numeric)
end

function umfpack_zl_get_lunz(lnz, unz, n_row, n_col, nz_udiag, Numeric)
    ccall((:umfpack_zl_get_lunz, libumfpack), Cint, (Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Cvoid}), lnz, unz, n_row, n_col, nz_udiag, Numeric)
end

function umfpack_di_get_numeric(Lp, Lj, Lx, Up, Ui, Ux, P, Q, Dx, do_recip, Rs, Numeric)
    ccall((:umfpack_di_get_numeric, libumfpack), Cint, (Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cvoid}), Lp, Lj, Lx, Up, Ui, Ux, P, Q, Dx, do_recip, Rs, Numeric)
end

function umfpack_dl_get_numeric(Lp, Lj, Lx, Up, Ui, Ux, P, Q, Dx, do_recip, Rs, Numeric)
    ccall((:umfpack_dl_get_numeric, libumfpack), Cint, (Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cvoid}), Lp, Lj, Lx, Up, Ui, Ux, P, Q, Dx, do_recip, Rs, Numeric)
end

function umfpack_zi_get_numeric(Lp, Lj, Lx, Lz, Up, Ui, Ux, Uz, P, Q, Dx, Dz, do_recip, Rs, Numeric)
    ccall((:umfpack_zi_get_numeric, libumfpack), Cint, (Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cvoid}), Lp, Lj, Lx, Lz, Up, Ui, Ux, Uz, P, Q, Dx, Dz, do_recip, Rs, Numeric)
end

function umfpack_zl_get_numeric(Lp, Lj, Lx, Lz, Up, Ui, Ux, Uz, P, Q, Dx, Dz, do_recip, Rs, Numeric)
    ccall((:umfpack_zl_get_numeric, libumfpack), Cint, (Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cvoid}), Lp, Lj, Lx, Lz, Up, Ui, Ux, Uz, P, Q, Dx, Dz, do_recip, Rs, Numeric)
end

function umfpack_di_get_symbolic(n_row, n_col, n1, nz, nfr, nchains, P, Q, Front_npivcol, Front_parent, Front_1strow, Front_leftmostdesc, Chain_start, Chain_maxrows, Chain_maxcols, Dmap, Symbolic)
    ccall((:umfpack_di_get_symbolic, libumfpack), Cint, (Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Cvoid}), n_row, n_col, n1, nz, nfr, nchains, P, Q, Front_npivcol, Front_parent, Front_1strow, Front_leftmostdesc, Chain_start, Chain_maxrows, Chain_maxcols, Dmap, Symbolic)
end

function umfpack_dl_get_symbolic(n_row, n_col, n1, nz, nfr, nchains, P, Q, Front_npivcol, Front_parent, Front_1strow, Front_leftmostdesc, Chain_start, Chain_maxrows, Chain_maxcols, Dmap, Symbolic)
    ccall((:umfpack_dl_get_symbolic, libumfpack), Cint, (Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Cvoid}), n_row, n_col, n1, nz, nfr, nchains, P, Q, Front_npivcol, Front_parent, Front_1strow, Front_leftmostdesc, Chain_start, Chain_maxrows, Chain_maxcols, Dmap, Symbolic)
end

function umfpack_zi_get_symbolic(n_row, n_col, n1, nz, nfr, nchains, P, Q, Front_npivcol, Front_parent, Front_1strow, Front_leftmostdesc, Chain_start, Chain_maxrows, Chain_maxcols, Dmap, Symbolic)
    ccall((:umfpack_zi_get_symbolic, libumfpack), Cint, (Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Cvoid}), n_row, n_col, n1, nz, nfr, nchains, P, Q, Front_npivcol, Front_parent, Front_1strow, Front_leftmostdesc, Chain_start, Chain_maxrows, Chain_maxcols, Dmap, Symbolic)
end

function umfpack_zl_get_symbolic(n_row, n_col, n1, nz, nfr, nchains, P, Q, Front_npivcol, Front_parent, Front_1strow, Front_leftmostdesc, Chain_start, Chain_maxrows, Chain_maxcols, Dmap, Symbolic)
    ccall((:umfpack_zl_get_symbolic, libumfpack), Cint, (Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Cvoid}), n_row, n_col, n1, nz, nfr, nchains, P, Q, Front_npivcol, Front_parent, Front_1strow, Front_leftmostdesc, Chain_start, Chain_maxrows, Chain_maxcols, Dmap, Symbolic)
end

function umfpack_di_save_numeric(Numeric, filename)
    ccall((:umfpack_di_save_numeric, libumfpack), Cint, (Ptr{Cvoid}, Ptr{Cchar}), Numeric, filename)
end

function umfpack_dl_save_numeric(Numeric, filename)
    ccall((:umfpack_dl_save_numeric, libumfpack), Cint, (Ptr{Cvoid}, Ptr{Cchar}), Numeric, filename)
end

function umfpack_zi_save_numeric(Numeric, filename)
    ccall((:umfpack_zi_save_numeric, libumfpack), Cint, (Ptr{Cvoid}, Ptr{Cchar}), Numeric, filename)
end

function umfpack_zl_save_numeric(Numeric, filename)
    ccall((:umfpack_zl_save_numeric, libumfpack), Cint, (Ptr{Cvoid}, Ptr{Cchar}), Numeric, filename)
end

function umfpack_di_load_numeric(Numeric, filename)
    ccall((:umfpack_di_load_numeric, libumfpack), Cint, (Ptr{Ptr{Cvoid}}, Ptr{Cchar}), Numeric, filename)
end

function umfpack_dl_load_numeric(Numeric, filename)
    ccall((:umfpack_dl_load_numeric, libumfpack), Cint, (Ptr{Ptr{Cvoid}}, Ptr{Cchar}), Numeric, filename)
end

function umfpack_zi_load_numeric(Numeric, filename)
    ccall((:umfpack_zi_load_numeric, libumfpack), Cint, (Ptr{Ptr{Cvoid}}, Ptr{Cchar}), Numeric, filename)
end

function umfpack_zl_load_numeric(Numeric, filename)
    ccall((:umfpack_zl_load_numeric, libumfpack), Cint, (Ptr{Ptr{Cvoid}}, Ptr{Cchar}), Numeric, filename)
end

function umfpack_di_copy_numeric(Numeric, Original)
    ccall((:umfpack_di_copy_numeric, libumfpack), Cint, (Ptr{Ptr{Cvoid}}, Ptr{Cvoid}), Numeric, Original)
end

function umfpack_dl_copy_numeric(Numeric, Original)
    ccall((:umfpack_dl_copy_numeric, libumfpack), Cint, (Ptr{Ptr{Cvoid}}, Ptr{Cvoid}), Numeric, Original)
end

function umfpack_zi_copy_numeric(Numeric, Original)
    ccall((:umfpack_zi_copy_numeric, libumfpack), Cint, (Ptr{Ptr{Cvoid}}, Ptr{Cvoid}), Numeric, Original)
end

function umfpack_zl_copy_numeric(Numeric, Original)
    ccall((:umfpack_zl_copy_numeric, libumfpack), Cint, (Ptr{Ptr{Cvoid}}, Ptr{Cvoid}), Numeric, Original)
end

function umfpack_di_serialize_numeric_size(blobsize, Numeric)
    ccall((:umfpack_di_serialize_numeric_size, libumfpack), Cint, (Ptr{Int64}, Ptr{Cvoid}), blobsize, Numeric)
end

function umfpack_dl_serialize_numeric_size(blobsize, Numeric)
    ccall((:umfpack_dl_serialize_numeric_size, libumfpack), Cint, (Ptr{Int64}, Ptr{Cvoid}), blobsize, Numeric)
end

function umfpack_zi_serialize_numeric_size(blobsize, Numeric)
    ccall((:umfpack_zi_serialize_numeric_size, libumfpack), Cint, (Ptr{Int64}, Ptr{Cvoid}), blobsize, Numeric)
end

function umfpack_zl_serialize_numeric_size(blobsize, Numeric)
    ccall((:umfpack_zl_serialize_numeric_size, libumfpack), Cint, (Ptr{Int64}, Ptr{Cvoid}), blobsize, Numeric)
end

function umfpack_di_serialize_numeric(blob, blobsize, Numeric)
    ccall((:umfpack_di_serialize_numeric, libumfpack), Cint, (Ptr{Int8}, Int64, Ptr{Cvoid}), blob, blobsize, Numeric)
end

function umfpack_dl_serialize_numeric(blob, blobsize, Numeric)
    ccall((:umfpack_dl_serialize_numeric, libumfpack), Cint, (Ptr{Int8}, Int64, Ptr{Cvoid}), blob, blobsize, Numeric)
end

function umfpack_zi_serialize_numeric(blob, blobsize, Numeric)
    ccall((:umfpack_zi_serialize_numeric, libumfpack), Cint, (Ptr{Int8}, Int64, Ptr{Cvoid}), blob, blobsize, Numeric)
end

function umfpack_zl_serialize_numeric(blob, blobsize, Numeric)
    ccall((:umfpack_zl_serialize_numeric, libumfpack), Cint, (Ptr{Int8}, Int64, Ptr{Cvoid}), blob, blobsize, Numeric)
end

function umfpack_di_deserialize_numeric(Numeric, blob, blobsize)
    ccall((:umfpack_di_deserialize_numeric, libumfpack), Cint, (Ptr{Ptr{Cvoid}}, Ptr{Int8}, Int64), Numeric, blob, blobsize)
end

function umfpack_dl_deserialize_numeric(Numeric, blob, blobsize)
    ccall((:umfpack_dl_deserialize_numeric, libumfpack), Cint, (Ptr{Ptr{Cvoid}}, Ptr{Int8}, Int64), Numeric, blob, blobsize)
end

function umfpack_zi_deserialize_numeric(Numeric, blob, blobsize)
    ccall((:umfpack_zi_deserialize_numeric, libumfpack), Cint, (Ptr{Ptr{Cvoid}}, Ptr{Int8}, Int64), Numeric, blob, blobsize)
end

function umfpack_zl_deserialize_numeric(Numeric, blob, blobsize)
    ccall((:umfpack_zl_deserialize_numeric, libumfpack), Cint, (Ptr{Ptr{Cvoid}}, Ptr{Int8}, Int64), Numeric, blob, blobsize)
end

function umfpack_di_save_symbolic(Symbolic, filename)
    ccall((:umfpack_di_save_symbolic, libumfpack), Cint, (Ptr{Cvoid}, Ptr{Cchar}), Symbolic, filename)
end

function umfpack_dl_save_symbolic(Symbolic, filename)
    ccall((:umfpack_dl_save_symbolic, libumfpack), Cint, (Ptr{Cvoid}, Ptr{Cchar}), Symbolic, filename)
end

function umfpack_zi_save_symbolic(Symbolic, filename)
    ccall((:umfpack_zi_save_symbolic, libumfpack), Cint, (Ptr{Cvoid}, Ptr{Cchar}), Symbolic, filename)
end

function umfpack_zl_save_symbolic(Symbolic, filename)
    ccall((:umfpack_zl_save_symbolic, libumfpack), Cint, (Ptr{Cvoid}, Ptr{Cchar}), Symbolic, filename)
end

function umfpack_di_load_symbolic(Symbolic, filename)
    ccall((:umfpack_di_load_symbolic, libumfpack), Cint, (Ptr{Ptr{Cvoid}}, Ptr{Cchar}), Symbolic, filename)
end

function umfpack_dl_load_symbolic(Symbolic, filename)
    ccall((:umfpack_dl_load_symbolic, libumfpack), Cint, (Ptr{Ptr{Cvoid}}, Ptr{Cchar}), Symbolic, filename)
end

function umfpack_zi_load_symbolic(Symbolic, filename)
    ccall((:umfpack_zi_load_symbolic, libumfpack), Cint, (Ptr{Ptr{Cvoid}}, Ptr{Cchar}), Symbolic, filename)
end

function umfpack_zl_load_symbolic(Symbolic, filename)
    ccall((:umfpack_zl_load_symbolic, libumfpack), Cint, (Ptr{Ptr{Cvoid}}, Ptr{Cchar}), Symbolic, filename)
end

function umfpack_di_copy_symbolic(Symbolic, Original)
    ccall((:umfpack_di_copy_symbolic, libumfpack), Cint, (Ptr{Ptr{Cvoid}}, Ptr{Cvoid}), Symbolic, Original)
end

function umfpack_dl_copy_symbolic(Symbolic, Original)
    ccall((:umfpack_dl_copy_symbolic, libumfpack), Cint, (Ptr{Ptr{Cvoid}}, Ptr{Cvoid}), Symbolic, Original)
end

function umfpack_zi_copy_symbolic(Symbolic, Original)
    ccall((:umfpack_zi_copy_symbolic, libumfpack), Cint, (Ptr{Ptr{Cvoid}}, Ptr{Cvoid}), Symbolic, Original)
end

function umfpack_zl_copy_symbolic(Symbolic, Original)
    ccall((:umfpack_zl_copy_symbolic, libumfpack), Cint, (Ptr{Ptr{Cvoid}}, Ptr{Cvoid}), Symbolic, Original)
end

function umfpack_di_serialize_symbolic_size(blobsize, Symbolic)
    ccall((:umfpack_di_serialize_symbolic_size, libumfpack), Cint, (Ptr{Int64}, Ptr{Cvoid}), blobsize, Symbolic)
end

function umfpack_dl_serialize_symbolic_size(blobsize, Symbolic)
    ccall((:umfpack_dl_serialize_symbolic_size, libumfpack), Cint, (Ptr{Int64}, Ptr{Cvoid}), blobsize, Symbolic)
end

function umfpack_zi_serialize_symbolic_size(blobsize, Symbolic)
    ccall((:umfpack_zi_serialize_symbolic_size, libumfpack), Cint, (Ptr{Int64}, Ptr{Cvoid}), blobsize, Symbolic)
end

function umfpack_zl_serialize_symbolic_size(blobsize, Symbolic)
    ccall((:umfpack_zl_serialize_symbolic_size, libumfpack), Cint, (Ptr{Int64}, Ptr{Cvoid}), blobsize, Symbolic)
end

function umfpack_di_serialize_symbolic(blob, blobsize, Symbolic)
    ccall((:umfpack_di_serialize_symbolic, libumfpack), Cint, (Ptr{Int8}, Int64, Ptr{Cvoid}), blob, blobsize, Symbolic)
end

function umfpack_dl_serialize_symbolic(blob, blobsize, Symbolic)
    ccall((:umfpack_dl_serialize_symbolic, libumfpack), Cint, (Ptr{Int8}, Int64, Ptr{Cvoid}), blob, blobsize, Symbolic)
end

function umfpack_zi_serialize_symbolic(blob, blobsize, Symbolic)
    ccall((:umfpack_zi_serialize_symbolic, libumfpack), Cint, (Ptr{Int8}, Int64, Ptr{Cvoid}), blob, blobsize, Symbolic)
end

function umfpack_zl_serialize_symbolic(blob, blobsize, Symbolic)
    ccall((:umfpack_zl_serialize_symbolic, libumfpack), Cint, (Ptr{Int8}, Int64, Ptr{Cvoid}), blob, blobsize, Symbolic)
end

function umfpack_di_deserialize_symbolic(Symbolic, blob, blobsize)
    ccall((:umfpack_di_deserialize_symbolic, libumfpack), Cint, (Ptr{Ptr{Cvoid}}, Ptr{Int8}, Int64), Symbolic, blob, blobsize)
end

function umfpack_dl_deserialize_symbolic(Symbolic, blob, blobsize)
    ccall((:umfpack_dl_deserialize_symbolic, libumfpack), Cint, (Ptr{Ptr{Cvoid}}, Ptr{Int8}, Int64), Symbolic, blob, blobsize)
end

function umfpack_zi_deserialize_symbolic(Symbolic, blob, blobsize)
    ccall((:umfpack_zi_deserialize_symbolic, libumfpack), Cint, (Ptr{Ptr{Cvoid}}, Ptr{Int8}, Int64), Symbolic, blob, blobsize)
end

function umfpack_zl_deserialize_symbolic(Symbolic, blob, blobsize)
    ccall((:umfpack_zl_deserialize_symbolic, libumfpack), Cint, (Ptr{Ptr{Cvoid}}, Ptr{Int8}, Int64), Symbolic, blob, blobsize)
end

function umfpack_di_get_determinant(Mx, Ex, Numeric, User_Info)
    ccall((:umfpack_di_get_determinant, libumfpack), Cint, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}, Ptr{Cdouble}), Mx, Ex, Numeric, User_Info)
end

function umfpack_dl_get_determinant(Mx, Ex, Numeric, User_Info)
    ccall((:umfpack_dl_get_determinant, libumfpack), Cint, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}, Ptr{Cdouble}), Mx, Ex, Numeric, User_Info)
end

function umfpack_zi_get_determinant(Mx, Mz, Ex, Numeric, User_Info)
    ccall((:umfpack_zi_get_determinant, libumfpack), Cint, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}, Ptr{Cdouble}), Mx, Mz, Ex, Numeric, User_Info)
end

function umfpack_zl_get_determinant(Mx, Mz, Ex, Numeric, User_Info)
    ccall((:umfpack_zl_get_determinant, libumfpack), Cint, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}, Ptr{Cdouble}), Mx, Mz, Ex, Numeric, User_Info)
end

function umfpack_di_report_status(Control, status)
    ccall((:umfpack_di_report_status, libumfpack), Cvoid, (Ptr{Cdouble}, Cint), Control, status)
end

function umfpack_dl_report_status(Control, status)
    ccall((:umfpack_dl_report_status, libumfpack), Cvoid, (Ptr{Cdouble}, Cint), Control, status)
end

function umfpack_zi_report_status(Control, status)
    ccall((:umfpack_zi_report_status, libumfpack), Cvoid, (Ptr{Cdouble}, Cint), Control, status)
end

function umfpack_zl_report_status(Control, status)
    ccall((:umfpack_zl_report_status, libumfpack), Cvoid, (Ptr{Cdouble}, Cint), Control, status)
end

function umfpack_di_report_info(Control, Info)
    ccall((:umfpack_di_report_info, libumfpack), Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}), Control, Info)
end

function umfpack_dl_report_info(Control, Info)
    ccall((:umfpack_dl_report_info, libumfpack), Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}), Control, Info)
end

function umfpack_zi_report_info(Control, Info)
    ccall((:umfpack_zi_report_info, libumfpack), Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}), Control, Info)
end

function umfpack_zl_report_info(Control, Info)
    ccall((:umfpack_zl_report_info, libumfpack), Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}), Control, Info)
end

function umfpack_di_report_control(Control)
    ccall((:umfpack_di_report_control, libumfpack), Cvoid, (Ptr{Cdouble},), Control)
end

function umfpack_dl_report_control(Control)
    ccall((:umfpack_dl_report_control, libumfpack), Cvoid, (Ptr{Cdouble},), Control)
end

function umfpack_zi_report_control(Control)
    ccall((:umfpack_zi_report_control, libumfpack), Cvoid, (Ptr{Cdouble},), Control)
end

function umfpack_zl_report_control(Control)
    ccall((:umfpack_zl_report_control, libumfpack), Cvoid, (Ptr{Cdouble},), Control)
end

function umfpack_di_report_matrix(n_row, n_col, Ap, Ai, Ax, col_form, Control)
    ccall((:umfpack_di_report_matrix, libumfpack), Cint, (Int32, Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Cint, Ptr{Cdouble}), n_row, n_col, Ap, Ai, Ax, col_form, Control)
end

function umfpack_dl_report_matrix(n_row, n_col, Ap, Ai, Ax, col_form, Control)
    ccall((:umfpack_dl_report_matrix, libumfpack), Cint, (Int64, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Cint, Ptr{Cdouble}), n_row, n_col, Ap, Ai, Ax, col_form, Control)
end

function umfpack_zi_report_matrix(n_row, n_col, Ap, Ai, Ax, Az, col_form, Control)
    ccall((:umfpack_zi_report_matrix, libumfpack), Cint, (Int32, Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}), n_row, n_col, Ap, Ai, Ax, Az, col_form, Control)
end

function umfpack_zl_report_matrix(n_row, n_col, Ap, Ai, Ax, Az, col_form, Control)
    ccall((:umfpack_zl_report_matrix, libumfpack), Cint, (Int64, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}), n_row, n_col, Ap, Ai, Ax, Az, col_form, Control)
end

function umfpack_di_report_triplet(n_row, n_col, nz, Ti, Tj, Tx, Control)
    ccall((:umfpack_di_report_triplet, libumfpack), Cint, (Int32, Int32, Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble}), n_row, n_col, nz, Ti, Tj, Tx, Control)
end

function umfpack_dl_report_triplet(n_row, n_col, nz, Ti, Tj, Tx, Control)
    ccall((:umfpack_dl_report_triplet, libumfpack), Cint, (Int64, Int64, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cdouble}), n_row, n_col, nz, Ti, Tj, Tx, Control)
end

function umfpack_zi_report_triplet(n_row, n_col, nz, Ti, Tj, Tx, Tz, Control)
    ccall((:umfpack_zi_report_triplet, libumfpack), Cint, (Int32, Int32, Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), n_row, n_col, nz, Ti, Tj, Tx, Tz, Control)
end

function umfpack_zl_report_triplet(n_row, n_col, nz, Ti, Tj, Tx, Tz, Control)
    ccall((:umfpack_zl_report_triplet, libumfpack), Cint, (Int64, Int64, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), n_row, n_col, nz, Ti, Tj, Tx, Tz, Control)
end

function umfpack_di_report_vector(n, X, Control)
    ccall((:umfpack_di_report_vector, libumfpack), Cint, (Int32, Ptr{Cdouble}, Ptr{Cdouble}), n, X, Control)
end

function umfpack_dl_report_vector(n, X, Control)
    ccall((:umfpack_dl_report_vector, libumfpack), Cint, (Int64, Ptr{Cdouble}, Ptr{Cdouble}), n, X, Control)
end

function umfpack_zi_report_vector(n, Xx, Xz, Control)
    ccall((:umfpack_zi_report_vector, libumfpack), Cint, (Int32, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), n, Xx, Xz, Control)
end

function umfpack_zl_report_vector(n, Xx, Xz, Control)
    ccall((:umfpack_zl_report_vector, libumfpack), Cint, (Int64, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), n, Xx, Xz, Control)
end

function umfpack_di_report_symbolic(Symbolic, Control)
    ccall((:umfpack_di_report_symbolic, libumfpack), Cint, (Ptr{Cvoid}, Ptr{Cdouble}), Symbolic, Control)
end

function umfpack_dl_report_symbolic(Symbolic, Control)
    ccall((:umfpack_dl_report_symbolic, libumfpack), Cint, (Ptr{Cvoid}, Ptr{Cdouble}), Symbolic, Control)
end

function umfpack_zi_report_symbolic(Symbolic, Control)
    ccall((:umfpack_zi_report_symbolic, libumfpack), Cint, (Ptr{Cvoid}, Ptr{Cdouble}), Symbolic, Control)
end

function umfpack_zl_report_symbolic(Symbolic, Control)
    ccall((:umfpack_zl_report_symbolic, libumfpack), Cint, (Ptr{Cvoid}, Ptr{Cdouble}), Symbolic, Control)
end

function umfpack_di_report_numeric(Numeric, Control)
    ccall((:umfpack_di_report_numeric, libumfpack), Cint, (Ptr{Cvoid}, Ptr{Cdouble}), Numeric, Control)
end

function umfpack_dl_report_numeric(Numeric, Control)
    ccall((:umfpack_dl_report_numeric, libumfpack), Cint, (Ptr{Cvoid}, Ptr{Cdouble}), Numeric, Control)
end

function umfpack_zi_report_numeric(Numeric, Control)
    ccall((:umfpack_zi_report_numeric, libumfpack), Cint, (Ptr{Cvoid}, Ptr{Cdouble}), Numeric, Control)
end

function umfpack_zl_report_numeric(Numeric, Control)
    ccall((:umfpack_zl_report_numeric, libumfpack), Cint, (Ptr{Cvoid}, Ptr{Cdouble}), Numeric, Control)
end

function umfpack_di_report_perm(np, Perm, Control)
    ccall((:umfpack_di_report_perm, libumfpack), Cint, (Int32, Ptr{Int32}, Ptr{Cdouble}), np, Perm, Control)
end

function umfpack_dl_report_perm(np, Perm, Control)
    ccall((:umfpack_dl_report_perm, libumfpack), Cint, (Int64, Ptr{Int64}, Ptr{Cdouble}), np, Perm, Control)
end

function umfpack_zi_report_perm(np, Perm, Control)
    ccall((:umfpack_zi_report_perm, libumfpack), Cint, (Int32, Ptr{Int32}, Ptr{Cdouble}), np, Perm, Control)
end

function umfpack_zl_report_perm(np, Perm, Control)
    ccall((:umfpack_zl_report_perm, libumfpack), Cint, (Int64, Ptr{Int64}, Ptr{Cdouble}), np, Perm, Control)
end

function umfpack_timer()
    ccall((:umfpack_timer, libumfpack), Cdouble, ())
end

function umfpack_tic(stats)
    ccall((:umfpack_tic, libumfpack), Cvoid, (Ptr{Cdouble},), stats)
end

function umfpack_toc(stats)
    ccall((:umfpack_toc, libumfpack), Cvoid, (Ptr{Cdouble},), stats)
end

const CHOLMOD_DATE = "Jan 17, 2023"

const CHOLMOD_MAIN_VERSION = 4

const CHOLMOD_SUB_VERSION = 0

const CHOLMOD_SUBSUB_VERSION = 3

const _FILE_OFFSET_BITS = 64

const SUITESPARSE_OPENMP_MAX_THREADS = 1

const SUITESPARSE_OPENMP_GET_NUM_THREADS = 1

const SUITESPARSE_OPENMP_GET_WTIME = 0

const SUITESPARSE_OPENMP_GET_THREAD_ID = 0

const SUITESPARSE_COMPILER_NVCC = 0

const SUITESPARSE_COMPILER_ICX = 0

const SUITESPARSE_COMPILER_ICC = 0

const SUITESPARSE_COMPILER_CLANG = 0

const SUITESPARSE_COMPILER_GCC = 0

const SUITESPARSE_COMPILER_MSC = 0

const SUITESPARSE_COMPILER_XLC = 0

const SUITESPARSE_DATE = "Jan 20, 2023"

const SUITESPARSE_MAIN_VERSION = 7

const SUITESPARSE_SUB_VERSION = 0

const SUITESPARSE_SUBSUB_VERSION = 1

SUITESPARSE_VER_CODE(main, sub) = main * 1000 + sub

const SUITESPARSE_VERSION = SUITESPARSE_VER_CODE(SUITESPARSE_MAIN_VERSION, SUITESPARSE_SUB_VERSION)

CHOLMOD_VER_CODE(main, sub) = main * 1000 + sub

const CHOLMOD_VERSION = CHOLMOD_VER_CODE(CHOLMOD_MAIN_VERSION, CHOLMOD_SUB_VERSION)

const CHOLMOD_DEVICE_SUPERNODE_BUFFERS = 6

const CHOLMOD_HOST_SUPERNODE_BUFFERS = 8

const CHOLMOD_DEVICE_STREAMS = 2

const CHOLMOD_COMMON = 0

const CHOLMOD_SPARSE = 1

const CHOLMOD_FACTOR = 2

const CHOLMOD_DENSE = 3

const CHOLMOD_TRIPLET = 4

const CHOLMOD_INT = 0

const CHOLMOD_INTLONG = 1

const CHOLMOD_LONG = 2

const CHOLMOD_DOUBLE = 0

const CHOLMOD_SINGLE = 1

const CHOLMOD_PATTERN = 0

const CHOLMOD_REAL = 1

const CHOLMOD_COMPLEX = 2

const CHOLMOD_ZOMPLEX = 3

const CHOLMOD_MAXMETHODS = 9

const CHOLMOD_OK = 0

const CHOLMOD_NOT_INSTALLED = -1

const CHOLMOD_OUT_OF_MEMORY = -2

const CHOLMOD_TOO_LARGE = -3

const CHOLMOD_INVALID = -4

const CHOLMOD_GPU_PROBLEM = -5

const CHOLMOD_NOT_POSDEF = 1

const CHOLMOD_DSMALL = 2

const CHOLMOD_NATURAL = 0

const CHOLMOD_GIVEN = 1

const CHOLMOD_AMD = 2

const CHOLMOD_METIS = 3

const CHOLMOD_NESDIS = 4

const CHOLMOD_COLAMD = 5

const CHOLMOD_POSTORDERED = 6

const CHOLMOD_SIMPLICIAL = 0

const CHOLMOD_AUTO = 1

const CHOLMOD_SUPERNODAL = 2

const CHOLMOD_ANALYZE_FOR_SPQR = 0

const CHOLMOD_ANALYZE_FOR_CHOLESKY = 1

const CHOLMOD_ANALYZE_FOR_SPQRGPU = 2

const CHOLMOD_MM_RECTANGULAR = 1

const CHOLMOD_MM_UNSYMMETRIC = 2

const CHOLMOD_MM_SYMMETRIC = 3

const CHOLMOD_MM_HERMITIAN = 4

const CHOLMOD_MM_SKEW_SYMMETRIC = 5

const CHOLMOD_MM_SYMMETRIC_POSDIAG = 6

const CHOLMOD_MM_HERMITIAN_POSDIAG = 7

const CHOLMOD_A = 0

const CHOLMOD_LDLt = 1

const CHOLMOD_LD = 2

const CHOLMOD_DLt = 3

const CHOLMOD_L = 4

const CHOLMOD_Lt = 5

const CHOLMOD_D = 6

const CHOLMOD_P = 7

const CHOLMOD_Pt = 8

const CHOLMOD_SCALAR = 0

const CHOLMOD_ROW = 1

const CHOLMOD_COL = 2

const CHOLMOD_SYM = 3

const SPQR_ORDERING_FIXED = 0

const SPQR_ORDERING_NATURAL = 1

const SPQR_ORDERING_COLAMD = 2

const SPQR_ORDERING_GIVEN = 3

const SPQR_ORDERING_CHOLMOD = 4

const SPQR_ORDERING_AMD = 5

const SPQR_ORDERING_METIS = 6

const SPQR_ORDERING_DEFAULT = 7

const SPQR_ORDERING_BEST = 8

const SPQR_ORDERING_BESTAMD = 9

const SPQR_DEFAULT_TOL = -2

const SPQR_NO_TOL = -1

const SPQR_QTX = 0

const SPQR_QX = 1

const SPQR_XQT = 2

const SPQR_XQ = 3

const SPQR_RX_EQUALS_B = 0

const SPQR_RETX_EQUALS_B = 1

const SPQR_RTX_EQUALS_B = 2

const SPQR_RTX_EQUALS_ETB = 3

const SPQR_DATE = "Jan 17, 2023"

const SPQR_MAIN_VERSION = 3

const SPQR_SUB_VERSION = 0

const SPQR_SUBSUB_VERSION = 3

SPQR_VER_CODE(main, sub) = main * 1000 + sub

const SPQR_VERSION = SPQR_VER_CODE(SPQR_MAIN_VERSION, SPQR_SUB_VERSION)

const Complex = Float64

const AMD_CONTROL = 5

const AMD_INFO = 20

const AMD_DENSE = 0

const AMD_AGGRESSIVE = 1

const AMD_DEFAULT_DENSE = 10.0

const AMD_DEFAULT_AGGRESSIVE = 1

const AMD_STATUS = 0

const AMD_N = 1

const AMD_NZ = 2

const AMD_SYMMETRY = 3

const AMD_NZDIAG = 4

const AMD_NZ_A_PLUS_AT = 5

const AMD_NDENSE = 6

const AMD_MEMORY = 7

const AMD_NCMPA = 8

const AMD_LNZ = 9

const AMD_NDIV = 10

const AMD_NMULTSUBS_LDL = 11

const AMD_NMULTSUBS_LU = 12

const AMD_DMAX = 13

const AMD_OK = 0

const AMD_OUT_OF_MEMORY = -1

const AMD_INVALID = -2

const AMD_OK_BUT_JUMBLED = 1

const AMD_DATE = "Jan 17, 2023"

const AMD_MAIN_VERSION = 3

const AMD_SUB_VERSION = 0

const AMD_SUBSUB_VERSION = 3

AMD_VERSION_CODE(main, sub) = main * 1000 + sub

const AMD_VERSION = AMD_VERSION_CODE(AMD_MAIN_VERSION, AMD_SUB_VERSION)

const COLAMD_DATE = "Jan 17, 2023"

const COLAMD_MAIN_VERSION = 3

const COLAMD_SUB_VERSION = 0

const COLAMD_SUBSUB_VERSION = 3

COLAMD_VERSION_CODE(main, sub) = main * 1000 + sub

const COLAMD_VERSION = COLAMD_VERSION_CODE(COLAMD_MAIN_VERSION, COLAMD_SUB_VERSION)

const COLAMD_KNOBS = 20

const COLAMD_STATS = 20

const COLAMD_DENSE_ROW = 0

const COLAMD_DENSE_COL = 1

const COLAMD_AGGRESSIVE = 2

const COLAMD_DEFRAG_COUNT = 2

const COLAMD_STATUS = 3

const COLAMD_INFO1 = 4

const COLAMD_INFO2 = 5

const COLAMD_INFO3 = 6

const COLAMD_OK = 0

const COLAMD_OK_BUT_JUMBLED = 1

const COLAMD_ERROR_A_not_present = -1

const COLAMD_ERROR_p_not_present = -2

const COLAMD_ERROR_nrow_negative = -3

const COLAMD_ERROR_ncol_negative = -4

const COLAMD_ERROR_nnz_negative = -5

const COLAMD_ERROR_p0_nonzero = -6

const COLAMD_ERROR_A_too_small = -7

const COLAMD_ERROR_col_length_negative = -8

const COLAMD_ERROR_row_index_out_of_bounds = -9

const COLAMD_ERROR_out_of_memory = -10

const COLAMD_ERROR_internal_error = -999

const CCOLAMD_DATE = "Jan 17, 2023"

const CCOLAMD_MAIN_VERSION = 3

const CCOLAMD_SUB_VERSION = 0

const CCOLAMD_SUBSUB_VERSION = 3

CCOLAMD_VERSION_CODE(main, sub) = main * 1000 + sub

const CCOLAMD_VERSION = CCOLAMD_VERSION_CODE(CCOLAMD_MAIN_VERSION, CCOLAMD_SUB_VERSION)

const CCOLAMD_KNOBS = 20

const CCOLAMD_STATS = 20

const CCOLAMD_DENSE_ROW = 0

const CCOLAMD_DENSE_COL = 1

const CCOLAMD_AGGRESSIVE = 2

const CCOLAMD_LU = 3

const CCOLAMD_DEFRAG_COUNT = 2

const CCOLAMD_STATUS = 3

const CCOLAMD_INFO1 = 4

const CCOLAMD_INFO2 = 5

const CCOLAMD_INFO3 = 6

const CCOLAMD_EMPTY_ROW = 7

const CCOLAMD_EMPTY_COL = 8

const CCOLAMD_NEWLY_EMPTY_ROW = 9

const CCOLAMD_NEWLY_EMPTY_COL = 10

const CCOLAMD_OK = 0

const CCOLAMD_OK_BUT_JUMBLED = 1

const CCOLAMD_ERROR_A_not_present = -1

const CCOLAMD_ERROR_p_not_present = -2

const CCOLAMD_ERROR_nrow_negative = -3

const CCOLAMD_ERROR_ncol_negative = -4

const CCOLAMD_ERROR_nnz_negative = -5

const CCOLAMD_ERROR_p0_nonzero = -6

const CCOLAMD_ERROR_A_too_small = -7

const CCOLAMD_ERROR_col_length_negative = -8

const CCOLAMD_ERROR_row_index_out_of_bounds = -9

const CCOLAMD_ERROR_out_of_memory = -10

const CCOLAMD_ERROR_invalid_cmember = -11

const CCOLAMD_ERROR_internal_error = -999

const UMFPACK_INFO = 90

const UMFPACK_CONTROL = 20

# Skipping MacroDefinition: UMFPACK_COPYRIGHT \
#"UMFPACK:  Copyright (c) 2005-2023 by Timothy A. Davis.  All Rights Reserved.\n"

# Skipping MacroDefinition: UMFPACK_LICENSE_PART1 \
#"\nUMFPACK License: SPDX-License-Identifier: GPL-2.0+\n" \
#"   UMFPACK is available under alternate licenses,\n" \
#"   contact T. Davis for details.\n"

const UMFPACK_LICENSE_PART2 = "\n"

# Skipping MacroDefinition: UMFPACK_LICENSE_PART3 \
#"\n" \
#"Availability: http://www.suitesparse.com" \
#"\n"

const UMFPACK_DATE = "Jan 17, 2023"

const UMFPACK_MAIN_VERSION = 6

const UMFPACK_SUB_VERSION = 1

const UMFPACK_SUBSUB_VERSION = 0

UMFPACK_VER_CODE(main, sub) = main * 1000 + sub

const UMFPACK_VER = UMFPACK_VER_CODE(UMFPACK_MAIN_VERSION, UMFPACK_SUB_VERSION)

const UMFPACK_STATUS = 0

const UMFPACK_NROW = 1

const UMFPACK_NCOL = 16

const UMFPACK_NZ = 2

const UMFPACK_SIZE_OF_UNIT = 3

const UMFPACK_SIZE_OF_INT = 4

const UMFPACK_SIZE_OF_LONG = 5

const UMFPACK_SIZE_OF_POINTER = 6

const UMFPACK_SIZE_OF_ENTRY = 7

const UMFPACK_NDENSE_ROW = 8

const UMFPACK_NEMPTY_ROW = 9

const UMFPACK_NDENSE_COL = 10

const UMFPACK_NEMPTY_COL = 11

const UMFPACK_SYMBOLIC_DEFRAG = 12

const UMFPACK_SYMBOLIC_PEAK_MEMORY = 13

const UMFPACK_SYMBOLIC_SIZE = 14

const UMFPACK_SYMBOLIC_TIME = 15

const UMFPACK_SYMBOLIC_WALLTIME = 17

const UMFPACK_STRATEGY_USED = 18

const UMFPACK_ORDERING_USED = 19

const UMFPACK_QFIXED = 31

const UMFPACK_DIAG_PREFERRED = 32

const UMFPACK_PATTERN_SYMMETRY = 33

const UMFPACK_NZ_A_PLUS_AT = 34

const UMFPACK_NZDIAG = 35

const UMFPACK_SYMMETRIC_LUNZ = 36

const UMFPACK_SYMMETRIC_FLOPS = 37

const UMFPACK_SYMMETRIC_NDENSE = 38

const UMFPACK_SYMMETRIC_DMAX = 39

const UMFPACK_COL_SINGLETONS = 56

const UMFPACK_ROW_SINGLETONS = 57

const UMFPACK_N2 = 58

const UMFPACK_S_SYMMETRIC = 59

const UMFPACK_NUMERIC_SIZE_ESTIMATE = 20

const UMFPACK_PEAK_MEMORY_ESTIMATE = 21

const UMFPACK_FLOPS_ESTIMATE = 22

const UMFPACK_LNZ_ESTIMATE = 23

const UMFPACK_UNZ_ESTIMATE = 24

const UMFPACK_VARIABLE_INIT_ESTIMATE = 25

const UMFPACK_VARIABLE_PEAK_ESTIMATE = 26

const UMFPACK_VARIABLE_FINAL_ESTIMATE = 27

const UMFPACK_MAX_FRONT_SIZE_ESTIMATE = 28

const UMFPACK_MAX_FRONT_NROWS_ESTIMATE = 29

const UMFPACK_MAX_FRONT_NCOLS_ESTIMATE = 30

const UMFPACK_NUMERIC_SIZE = 40

const UMFPACK_PEAK_MEMORY = 41

const UMFPACK_FLOPS = 42

const UMFPACK_LNZ = 43

const UMFPACK_UNZ = 44

const UMFPACK_VARIABLE_INIT = 45

const UMFPACK_VARIABLE_PEAK = 46

const UMFPACK_VARIABLE_FINAL = 47

const UMFPACK_MAX_FRONT_SIZE = 48

const UMFPACK_MAX_FRONT_NROWS = 49

const UMFPACK_MAX_FRONT_NCOLS = 50

const UMFPACK_NUMERIC_DEFRAG = 60

const UMFPACK_NUMERIC_REALLOC = 61

const UMFPACK_NUMERIC_COSTLY_REALLOC = 62

const UMFPACK_COMPRESSED_PATTERN = 63

const UMFPACK_LU_ENTRIES = 64

const UMFPACK_NUMERIC_TIME = 65

const UMFPACK_UDIAG_NZ = 66

const UMFPACK_RCOND = 67

const UMFPACK_WAS_SCALED = 68

const UMFPACK_RSMIN = 69

const UMFPACK_RSMAX = 70

const UMFPACK_UMIN = 71

const UMFPACK_UMAX = 72

const UMFPACK_ALLOC_INIT_USED = 73

const UMFPACK_FORCED_UPDATES = 74

const UMFPACK_NUMERIC_WALLTIME = 75

const UMFPACK_NOFF_DIAG = 76

const UMFPACK_ALL_LNZ = 77

const UMFPACK_ALL_UNZ = 78

const UMFPACK_NZDROPPED = 79

const UMFPACK_IR_TAKEN = 80

const UMFPACK_IR_ATTEMPTED = 81

const UMFPACK_OMEGA1 = 82

const UMFPACK_OMEGA2 = 83

const UMFPACK_SOLVE_FLOPS = 84

const UMFPACK_SOLVE_TIME = 85

const UMFPACK_SOLVE_WALLTIME = 86

const UMFPACK_PRL = 0

const UMFPACK_DENSE_ROW = 1

const UMFPACK_DENSE_COL = 2

const UMFPACK_BLOCK_SIZE = 4

const UMFPACK_STRATEGY = 5

const UMFPACK_ORDERING = 10

const UMFPACK_FIXQ = 13

const UMFPACK_AMD_DENSE = 14

const UMFPACK_AGGRESSIVE = 19

const UMFPACK_SINGLETONS = 11

const UMFPACK_PIVOT_TOLERANCE = 3

const UMFPACK_ALLOC_INIT = 6

const UMFPACK_SYM_PIVOT_TOLERANCE = 15

const UMFPACK_SCALE = 16

const UMFPACK_FRONT_ALLOC_INIT = 17

const UMFPACK_DROPTOL = 18

const UMFPACK_IRSTEP = 7

const UMFPACK_COMPILED_WITH_BLAS = 8

const UMFPACK_STRATEGY_THRESH_SYM = 9

const UMFPACK_STRATEGY_THRESH_NNZDIAG = 12

const UMFPACK_STRATEGY_AUTO = 0

const UMFPACK_STRATEGY_UNSYMMETRIC = 1

const UMFPACK_STRATEGY_OBSOLETE = 2

const UMFPACK_STRATEGY_SYMMETRIC = 3

const UMFPACK_SCALE_NONE = 0

const UMFPACK_SCALE_SUM = 1

const UMFPACK_SCALE_MAX = 2

const UMFPACK_ORDERING_CHOLMOD = 0

const UMFPACK_ORDERING_AMD = 1

const UMFPACK_ORDERING_GIVEN = 2

const UMFPACK_ORDERING_METIS = 3

const UMFPACK_ORDERING_BEST = 4

const UMFPACK_ORDERING_NONE = 5

const UMFPACK_ORDERING_USER = 6

const UMFPACK_ORDERING_METIS_GUARD = 7

const UMFPACK_DEFAULT_PRL = 1

const UMFPACK_DEFAULT_DENSE_ROW = 0.2

const UMFPACK_DEFAULT_DENSE_COL = 0.2

const UMFPACK_DEFAULT_PIVOT_TOLERANCE = 0.1

const UMFPACK_DEFAULT_SYM_PIVOT_TOLERANCE = 0.001

const UMFPACK_DEFAULT_BLOCK_SIZE = 32

const UMFPACK_DEFAULT_ALLOC_INIT = 0.7

const UMFPACK_DEFAULT_FRONT_ALLOC_INIT = 0.5

const UMFPACK_DEFAULT_IRSTEP = 2

const UMFPACK_DEFAULT_SCALE = UMFPACK_SCALE_SUM

const UMFPACK_DEFAULT_STRATEGY = UMFPACK_STRATEGY_AUTO

const UMFPACK_DEFAULT_AMD_DENSE = AMD_DEFAULT_DENSE

const UMFPACK_DEFAULT_FIXQ = 0

const UMFPACK_DEFAULT_AGGRESSIVE = 1

const UMFPACK_DEFAULT_DROPTOL = 0

const UMFPACK_DEFAULT_ORDERING = UMFPACK_ORDERING_AMD

const UMFPACK_DEFAULT_SINGLETONS = TRUE

const UMFPACK_DEFAULT_STRATEGY_THRESH_SYM = 0.3

const UMFPACK_DEFAULT_STRATEGY_THRESH_NNZDIAG = 0.9

const UMFPACK_OK = 0

const UMFPACK_WARNING_singular_matrix = 1

const UMFPACK_WARNING_determinant_underflow = 2

const UMFPACK_WARNING_determinant_overflow = 3

const UMFPACK_ERROR_out_of_memory = -1

const UMFPACK_ERROR_invalid_Numeric_object = -3

const UMFPACK_ERROR_invalid_Symbolic_object = -4

const UMFPACK_ERROR_argument_missing = -5

const UMFPACK_ERROR_n_nonpositive = -6

const UMFPACK_ERROR_invalid_matrix = -8

const UMFPACK_ERROR_different_pattern = -11

const UMFPACK_ERROR_invalid_system = -13

const UMFPACK_ERROR_invalid_permutation = -15

const UMFPACK_ERROR_internal_error = -911

const UMFPACK_ERROR_file_IO = -17

const UMFPACK_ERROR_ordering_failed = -18

const UMFPACK_ERROR_invalid_blob = -19

const UMFPACK_A = 0

const UMFPACK_At = 1

const UMFPACK_Aat = 2

const UMFPACK_Pt_L = 3

const UMFPACK_L = 4

const UMFPACK_Lt_P = 5

const UMFPACK_Lat_P = 6

const UMFPACK_Lt = 7

const UMFPACK_Lat = 8

const UMFPACK_U_Qt = 9

const UMFPACK_U = 10

const UMFPACK_Q_Ut = 11

const UMFPACK_Q_Uat = 12

const UMFPACK_Ut = 13

const UMFPACK_Uat = 14

