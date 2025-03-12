function SuiteSparse_time()
    @ccall libsuitesparseconfig.SuiteSparse_time()::Cdouble
end

function SuiteSparse_config_printf_func_get()
    @ccall libsuitesparseconfig.SuiteSparse_config_printf_func_get()::Ptr{Cvoid}
end

function cholmod_version(version)
    @ccall libcholmod.cholmod_version(version::Ptr{Cint})::Cint
end

function cholmod_l_version(version)
    @ccall libcholmod.cholmod_l_version(version::Ptr{Cint})::Cint
end

function SuiteSparse_config_malloc_func_get()
    @ccall libsuitesparseconfig.SuiteSparse_config_malloc_func_get()::Ptr{Cvoid}
end

function SuiteSparse_config_calloc_func_get()
    @ccall libsuitesparseconfig.SuiteSparse_config_calloc_func_get()::Ptr{Cvoid}
end

function SuiteSparse_config_realloc_func_get()
    @ccall libsuitesparseconfig.SuiteSparse_config_realloc_func_get()::Ptr{Cvoid}
end

function SuiteSparse_config_free_func_get()
    @ccall libsuitesparseconfig.SuiteSparse_config_free_func_get()::Ptr{Cvoid}
end

function SuiteSparse_config_hypot_func_get()
    @ccall libsuitesparseconfig.SuiteSparse_config_hypot_func_get()::Ptr{Cvoid}
end

function SuiteSparse_config_divcomplex_func_get()
    @ccall libsuitesparseconfig.SuiteSparse_config_divcomplex_func_get()::Ptr{Cvoid}
end

function SuiteSparse_config_malloc_func_set(malloc_func)
    @ccall libsuitesparseconfig.SuiteSparse_config_malloc_func_set(malloc_func::Ptr{Cvoid})::Cvoid
end

function SuiteSparse_config_calloc_func_set(calloc_func)
    @ccall libsuitesparseconfig.SuiteSparse_config_calloc_func_set(calloc_func::Ptr{Cvoid})::Cvoid
end

function SuiteSparse_config_realloc_func_set(realloc_func)
    @ccall libsuitesparseconfig.SuiteSparse_config_realloc_func_set(realloc_func::Ptr{Cvoid})::Cvoid
end

function SuiteSparse_config_free_func_set(free_func)
    @ccall libsuitesparseconfig.SuiteSparse_config_free_func_set(free_func::Ptr{Cvoid})::Cvoid
end

function SuiteSparse_config_printf_func_set(printf_func)
    @ccall libsuitesparseconfig.SuiteSparse_config_printf_func_set(printf_func::Ptr{Cvoid})::Cvoid
end

function SuiteSparse_config_hypot_func_set(hypot_func)
    @ccall libsuitesparseconfig.SuiteSparse_config_hypot_func_set(hypot_func::Ptr{Cvoid})::Cvoid
end

function SuiteSparse_config_divcomplex_func_set(divcomplex_func)
    @ccall libsuitesparseconfig.SuiteSparse_config_divcomplex_func_set(divcomplex_func::Ptr{Cvoid})::Cvoid
end

function SuiteSparse_config_malloc(s)
    @ccall libsuitesparseconfig.SuiteSparse_config_malloc(s::Csize_t)::Ptr{Cvoid}
end

function SuiteSparse_config_calloc(n, s)
    @ccall libsuitesparseconfig.SuiteSparse_config_calloc(n::Csize_t,
                                                          s::Csize_t)::Ptr{Cvoid}
end

function SuiteSparse_config_realloc(arg1, s)
    @ccall libsuitesparseconfig.SuiteSparse_config_realloc(arg1::Ptr{Cvoid},
                                                           s::Csize_t)::Ptr{Cvoid}
end

function SuiteSparse_config_free(arg1)
    @ccall libsuitesparseconfig.SuiteSparse_config_free(arg1::Ptr{Cvoid})::Cvoid
end

function SuiteSparse_config_hypot(x, y)
    @ccall libsuitesparseconfig.SuiteSparse_config_hypot(x::Cdouble, y::Cdouble)::Cdouble
end

function SuiteSparse_config_divcomplex(xr, xi, yr, yi, zr, zi)
    @ccall libsuitesparseconfig.SuiteSparse_config_divcomplex(xr::Cdouble, xi::Cdouble,
                                                              yr::Cdouble, yi::Cdouble,
                                                              zr::Ptr{Cdouble},
                                                              zi::Ptr{Cdouble})::Cint
end

function SuiteSparse_start()
    @ccall libsuitesparseconfig.SuiteSparse_start()::Cvoid
end

function SuiteSparse_finish()
    @ccall libsuitesparseconfig.SuiteSparse_finish()::Cvoid
end

function SuiteSparse_malloc(nitems, size_of_item)
    @ccall libsuitesparseconfig.SuiteSparse_malloc(nitems::Csize_t,
                                                   size_of_item::Csize_t)::Ptr{Cvoid}
end

function SuiteSparse_calloc(nitems, size_of_item)
    @ccall libsuitesparseconfig.SuiteSparse_calloc(nitems::Csize_t,
                                                   size_of_item::Csize_t)::Ptr{Cvoid}
end

function SuiteSparse_realloc(nitems_new, nitems_old, size_of_item, p, ok)
    @ccall libsuitesparseconfig.SuiteSparse_realloc(nitems_new::Csize_t,
                                                    nitems_old::Csize_t,
                                                    size_of_item::Csize_t, p::Ptr{Cvoid},
                                                    ok::Ptr{Cint})::Ptr{Cvoid}
end

function SuiteSparse_free(p)
    @ccall libsuitesparseconfig.SuiteSparse_free(p::Ptr{Cvoid})::Ptr{Cvoid}
end

function SuiteSparse_tic(tic)
    @ccall libsuitesparseconfig.SuiteSparse_tic(tic::Ptr{Cdouble})::Cvoid
end

function SuiteSparse_toc(tic)
    @ccall libsuitesparseconfig.SuiteSparse_toc(tic::Ptr{Cdouble})::Cdouble
end

function SuiteSparse_hypot(x, y)
    @ccall libsuitesparseconfig.SuiteSparse_hypot(x::Cdouble, y::Cdouble)::Cdouble
end

function SuiteSparse_divcomplex(ar, ai, br, bi, cr, ci)
    @ccall libsuitesparseconfig.SuiteSparse_divcomplex(ar::Cdouble, ai::Cdouble,
                                                       br::Cdouble, bi::Cdouble,
                                                       cr::Ptr{Cdouble},
                                                       ci::Ptr{Cdouble})::Cint
end

function SuiteSparse_version(version)
    @ccall libsuitesparseconfig.SuiteSparse_version(version::Ptr{Cint})::Cint
end

function SuiteSparse_BLAS_library()
    @ccall libsuitesparseconfig.SuiteSparse_BLAS_library()::Ptr{Cchar}
end

function SuiteSparse_BLAS_integer_size()
    @ccall libsuitesparseconfig.SuiteSparse_BLAS_integer_size()::Csize_t
end

@enum cholmod_query_t::UInt32 begin
    CHOLMOD_QUERY_HAS_GPL = 0
    CHOLMOD_QUERY_HAS_CHECK = 1
    CHOLMOD_QUERY_HAS_CHOLESKY = 2
    CHOLMOD_QUERY_HAS_CAMD = 3
    CHOLMOD_QUERY_HAS_PARTITION = 4
    CHOLMOD_QUERY_HAS_MATRIXOPS = 5
    CHOLMOD_QUERY_HAS_MODIFY = 6
    CHOLMOD_QUERY_HAS_SUPERNODAL = 7
    CHOLMOD_QUERY_HAS_CUDA = 8
    CHOLMOD_QUERY_HAS_OPENMP = 9
end

function cholmod_query(feature)
    @ccall libcholmod.cholmod_query(feature::cholmod_query_t)::Bool
end

function cholmod_l_query(feature)
    @ccall libcholmod.cholmod_l_query(feature::cholmod_query_t)::Bool
end

struct cholmod_method_struct
    lnz::Cdouble
    fl::Cdouble
    prune_dense::Cdouble
    prune_dense2::Cdouble
    nd_oksep::Cdouble
    other_1::NTuple{4,Cdouble}
    nd_small::Csize_t
    other_2::NTuple{4,Cdouble}
    aggressive::Cint
    order_for_lu::Cint
    nd_compress::Cint
    nd_camd::Cint
    nd_components::Cint
    ordering::Cint
    other_3::NTuple{4,Csize_t}
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
    zrelax::NTuple{3,Cdouble}
    nrelax::NTuple{3,Csize_t}
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
    method::NTuple{10,cholmod_method_struct}
    postorder::Cint
    default_nesdis::Cint
    metis_memory::Cdouble
    metis_dswitch::Cdouble
    metis_nswitch::Csize_t
    nrow::Csize_t
    mark::Int64
    iworksize::Csize_t
    xworkbytes::Csize_t
    Flag::Ptr{Cvoid}
    Head::Ptr{Cvoid}
    Xwork::Ptr{Cvoid}
    Iwork::Ptr{Cvoid}
    itype::Cint
    other_5::Cint
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
    SPQR_istat::NTuple{8,Int64}
    nsbounds_hit::Cdouble
    sbound::Cfloat
    other_6::Cfloat
    useGPU::Cint
    maxGpuMemBytes::Csize_t
    maxGpuMemFraction::Cdouble
    gpuMemorySize::Csize_t
    gpuKernelTime::Cdouble
    gpuFlops::Int64
    gpuNumKernelLaunches::Cint
    cublasHandle::Ptr{Cvoid}
    gpuStream::NTuple{8,Ptr{Cvoid}}
    cublasEventPotrf::NTuple{3,Ptr{Cvoid}}
    updateCKernelsComplete::Ptr{Cvoid}
    updateCBuffersFree::NTuple{8,Ptr{Cvoid}}
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
    @ccall libcholmod.cholmod_start(Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_start(arg1)
    @ccall libcholmod.cholmod_l_start(arg1::Ptr{cholmod_common})::Cint
end

function cholmod_finish(Common)
    @ccall libcholmod.cholmod_finish(Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_finish(arg1)
    @ccall libcholmod.cholmod_l_finish(arg1::Ptr{cholmod_common})::Cint
end

function cholmod_defaults(Common)
    @ccall libcholmod.cholmod_defaults(Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_defaults(arg1)
    @ccall libcholmod.cholmod_l_defaults(arg1::Ptr{cholmod_common})::Cint
end

function cholmod_maxrank(n, Common)
    @ccall libcholmod.cholmod_maxrank(n::Csize_t, Common::Ptr{cholmod_common})::Csize_t
end

function cholmod_l_maxrank(arg1, arg2)
    @ccall libcholmod.cholmod_l_maxrank(arg1::Csize_t, arg2::Ptr{cholmod_common})::Csize_t
end

function cholmod_allocate_work(nrow, iworksize, xworksize, Common)
    @ccall libcholmod.cholmod_allocate_work(nrow::Csize_t, iworksize::Csize_t,
                                            xworksize::Csize_t,
                                            Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_allocate_work(arg1, arg2, arg3, arg4)
    @ccall libcholmod.cholmod_l_allocate_work(arg1::Csize_t, arg2::Csize_t, arg3::Csize_t,
                                              arg4::Ptr{cholmod_common})::Cint
end

function cholmod_alloc_work(nrow, iworksize, xworksize, dtype, Common)
    @ccall libcholmod.cholmod_alloc_work(nrow::Csize_t, iworksize::Csize_t,
                                         xworksize::Csize_t, dtype::Cint,
                                         Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_alloc_work(arg1, arg2, arg3, arg4, arg5)
    @ccall libcholmod.cholmod_l_alloc_work(arg1::Csize_t, arg2::Csize_t, arg3::Csize_t,
                                           arg4::Cint, arg5::Ptr{cholmod_common})::Cint
end

function cholmod_free_work(Common)
    @ccall libcholmod.cholmod_free_work(Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_free_work(arg1)
    @ccall libcholmod.cholmod_l_free_work(arg1::Ptr{cholmod_common})::Cint
end

function cholmod_clear_flag(Common)
    @ccall libcholmod.cholmod_clear_flag(Common::Ptr{cholmod_common})::Int64
end

function cholmod_l_clear_flag(arg1)
    @ccall libcholmod.cholmod_l_clear_flag(arg1::Ptr{cholmod_common})::Int64
end

function cholmod_error(status, file, line, message, Common)
    @ccall libcholmod.cholmod_error(status::Cint, file::Ptr{Cchar}, line::Cint,
                                    message::Ptr{Cchar}, Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_error(arg1, arg2, arg3, arg4, arg5)
    @ccall libcholmod.cholmod_l_error(arg1::Cint, arg2::Ptr{Cchar}, arg3::Cint,
                                      arg4::Ptr{Cchar}, arg5::Ptr{cholmod_common})::Cint
end

function cholmod_dbound(arg1, arg2)
    @ccall libcholmod.cholmod_dbound(arg1::Cdouble, arg2::Ptr{cholmod_common})::Cdouble
end

function cholmod_l_dbound(arg1, arg2)
    @ccall libcholmod.cholmod_l_dbound(arg1::Cdouble, arg2::Ptr{cholmod_common})::Cdouble
end

function cholmod_sbound(arg1, arg2)
    @ccall libcholmod.cholmod_sbound(arg1::Cfloat, arg2::Ptr{cholmod_common})::Cfloat
end

function cholmod_l_sbound(arg1, arg2)
    @ccall libcholmod.cholmod_l_sbound(arg1::Cfloat, arg2::Ptr{cholmod_common})::Cfloat
end

function cholmod_hypot(x, y)
    @ccall libcholmod.cholmod_hypot(x::Cdouble, y::Cdouble)::Cdouble
end

function cholmod_l_hypot(arg1, arg2)
    @ccall libcholmod.cholmod_l_hypot(arg1::Cdouble, arg2::Cdouble)::Cdouble
end

function cholmod_divcomplex(ar, ai, br, bi, cr, ci)
    @ccall libcholmod.cholmod_divcomplex(ar::Cdouble, ai::Cdouble, br::Cdouble, bi::Cdouble,
                                         cr::Ptr{Cdouble}, ci::Ptr{Cdouble})::Cint
end

function cholmod_l_divcomplex(arg1, arg2, arg3, arg4, arg5, arg6)
    @ccall libcholmod.cholmod_l_divcomplex(arg1::Cdouble, arg2::Cdouble, arg3::Cdouble,
                                           arg4::Cdouble, arg5::Ptr{Cdouble},
                                           arg6::Ptr{Cdouble})::Cint
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

function cholmod_allocate_sparse(nrow, ncol, nzmax, sorted, packed, stype, xdtype, Common)
    @ccall libcholmod.cholmod_allocate_sparse(nrow::Csize_t, ncol::Csize_t, nzmax::Csize_t,
                                              sorted::Cint, packed::Cint, stype::Cint,
                                              xdtype::Cint,
                                              Common::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_l_allocate_sparse(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
    @ccall libcholmod.cholmod_l_allocate_sparse(arg1::Csize_t, arg2::Csize_t, arg3::Csize_t,
                                                arg4::Cint, arg5::Cint, arg6::Cint,
                                                arg7::Cint,
                                                arg8::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_free_sparse(A, Common)
    @ccall libcholmod.cholmod_free_sparse(A::Ptr{Ptr{cholmod_sparse}},
                                          Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_free_sparse(arg1, arg2)
    @ccall libcholmod.cholmod_l_free_sparse(arg1::Ptr{Ptr{cholmod_sparse}},
                                            arg2::Ptr{cholmod_common})::Cint
end

function cholmod_reallocate_sparse(nznew, A, Common)
    @ccall libcholmod.cholmod_reallocate_sparse(nznew::Csize_t, A::Ptr{cholmod_sparse},
                                                Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_reallocate_sparse(arg1, arg2, arg3)
    @ccall libcholmod.cholmod_l_reallocate_sparse(arg1::Csize_t, arg2::Ptr{cholmod_sparse},
                                                  arg3::Ptr{cholmod_common})::Cint
end

function cholmod_nnz(A, Common)
    @ccall libcholmod.cholmod_nnz(A::Ptr{cholmod_sparse},
                                  Common::Ptr{cholmod_common})::Int64
end

function cholmod_l_nnz(arg1, arg2)
    @ccall libcholmod.cholmod_l_nnz(arg1::Ptr{cholmod_sparse},
                                    arg2::Ptr{cholmod_common})::Int64
end

function cholmod_speye(nrow, ncol, xdtype, Common)
    @ccall libcholmod.cholmod_speye(nrow::Csize_t, ncol::Csize_t, xdtype::Cint,
                                    Common::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_l_speye(arg1, arg2, arg3, arg4)
    @ccall libcholmod.cholmod_l_speye(arg1::Csize_t, arg2::Csize_t, arg3::Cint,
                                      arg4::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_spzeros(nrow, ncol, nzmax, xdtype, Common)
    @ccall libcholmod.cholmod_spzeros(nrow::Csize_t, ncol::Csize_t, nzmax::Csize_t,
                                      xdtype::Cint,
                                      Common::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_l_spzeros(arg1, arg2, arg3, arg4, arg5)
    @ccall libcholmod.cholmod_l_spzeros(arg1::Csize_t, arg2::Csize_t, arg3::Csize_t,
                                        arg4::Cint,
                                        arg5::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_transpose(A, mode, Common)
    @ccall libcholmod.cholmod_transpose(A::Ptr{cholmod_sparse}, mode::Cint,
                                        Common::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_l_transpose(arg1, arg2, arg3)
    @ccall libcholmod.cholmod_l_transpose(arg1::Ptr{cholmod_sparse}, arg2::Cint,
                                          arg3::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_transpose_unsym(A, mode, Perm, fset, fsize, C, Common)
    @ccall libcholmod.cholmod_transpose_unsym(A::Ptr{cholmod_sparse}, mode::Cint,
                                              Perm::Ptr{Int32}, fset::Ptr{Int32},
                                              fsize::Csize_t, C::Ptr{cholmod_sparse},
                                              Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_transpose_unsym(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    @ccall libcholmod.cholmod_l_transpose_unsym(arg1::Ptr{cholmod_sparse}, arg2::Cint,
                                                arg3::Ptr{Int64}, arg4::Ptr{Int64},
                                                arg5::Csize_t, arg6::Ptr{cholmod_sparse},
                                                arg7::Ptr{cholmod_common})::Cint
end

function cholmod_transpose_sym(A, mode, Perm, C, Common)
    @ccall libcholmod.cholmod_transpose_sym(A::Ptr{cholmod_sparse}, mode::Cint,
                                            Perm::Ptr{Int32}, C::Ptr{cholmod_sparse},
                                            Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_transpose_sym(arg1, arg2, arg3, arg4, arg5)
    @ccall libcholmod.cholmod_l_transpose_sym(arg1::Ptr{cholmod_sparse}, arg2::Cint,
                                              arg3::Ptr{Int64}, arg4::Ptr{cholmod_sparse},
                                              arg5::Ptr{cholmod_common})::Cint
end

function cholmod_ptranspose(A, mode, Perm, fset, fsize, Common)
    @ccall libcholmod.cholmod_ptranspose(A::Ptr{cholmod_sparse}, mode::Cint,
                                         Perm::Ptr{Int32}, fset::Ptr{Int32}, fsize::Csize_t,
                                         Common::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_l_ptranspose(arg1, arg2, arg3, arg4, arg5, arg6)
    @ccall libcholmod.cholmod_l_ptranspose(arg1::Ptr{cholmod_sparse}, arg2::Cint,
                                           arg3::Ptr{Int64}, arg4::Ptr{Int64},
                                           arg5::Csize_t,
                                           arg6::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_sort(A, Common)
    @ccall libcholmod.cholmod_sort(A::Ptr{cholmod_sparse},
                                   Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_sort(arg1, arg2)
    @ccall libcholmod.cholmod_l_sort(arg1::Ptr{cholmod_sparse},
                                     arg2::Ptr{cholmod_common})::Cint
end

function cholmod_band_nnz(A, k1, k2, ignore_diag, Common)
    @ccall libcholmod.cholmod_band_nnz(A::Ptr{cholmod_sparse}, k1::Int64, k2::Int64,
                                       ignore_diag::Bool,
                                       Common::Ptr{cholmod_common})::Int64
end

function cholmod_l_band_nnz(arg1, arg2, arg3, arg4, arg5)
    @ccall libcholmod.cholmod_l_band_nnz(arg1::Ptr{cholmod_sparse}, arg2::Int64,
                                         arg3::Int64, arg4::Bool,
                                         arg5::Ptr{cholmod_common})::Int64
end

function cholmod_band(A, k1, k2, mode, Common)
    @ccall libcholmod.cholmod_band(A::Ptr{cholmod_sparse}, k1::Int64, k2::Int64, mode::Cint,
                                   Common::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_l_band(arg1, arg2, arg3, arg4, arg5)
    @ccall libcholmod.cholmod_l_band(arg1::Ptr{cholmod_sparse}, arg2::Int64, arg3::Int64,
                                     arg4::Cint,
                                     arg5::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_band_inplace(k1, k2, mode, A, Common)
    @ccall libcholmod.cholmod_band_inplace(k1::Int64, k2::Int64, mode::Cint,
                                           A::Ptr{cholmod_sparse},
                                           Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_band_inplace(arg1, arg2, arg3, arg4, arg5)
    @ccall libcholmod.cholmod_l_band_inplace(arg1::Int64, arg2::Int64, arg3::Cint,
                                             arg4::Ptr{cholmod_sparse},
                                             arg5::Ptr{cholmod_common})::Cint
end

function cholmod_aat(A, fset, fsize, mode, Common)
    @ccall libcholmod.cholmod_aat(A::Ptr{cholmod_sparse}, fset::Ptr{Int32}, fsize::Csize_t,
                                  mode::Cint,
                                  Common::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_l_aat(arg1, arg2, arg3, arg4, arg5)
    @ccall libcholmod.cholmod_l_aat(arg1::Ptr{cholmod_sparse}, arg2::Ptr{Int64},
                                    arg3::Csize_t, arg4::Cint,
                                    arg5::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_copy_sparse(A, Common)
    @ccall libcholmod.cholmod_copy_sparse(A::Ptr{cholmod_sparse},
                                          Common::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_l_copy_sparse(arg1, arg2)
    @ccall libcholmod.cholmod_l_copy_sparse(arg1::Ptr{cholmod_sparse},
                                            arg2::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_copy(A, stype, mode, Common)
    @ccall libcholmod.cholmod_copy(A::Ptr{cholmod_sparse}, stype::Cint, mode::Cint,
                                   Common::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_l_copy(arg1, arg2, arg3, arg4)
    @ccall libcholmod.cholmod_l_copy(arg1::Ptr{cholmod_sparse}, arg2::Cint, arg3::Cint,
                                     arg4::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_add(A, B, alpha, beta, mode, sorted, Common)
    @ccall libcholmod.cholmod_add(A::Ptr{cholmod_sparse}, B::Ptr{cholmod_sparse},
                                  alpha::Ptr{Cdouble}, beta::Ptr{Cdouble}, mode::Cint,
                                  sorted::Cint,
                                  Common::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_l_add(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    @ccall libcholmod.cholmod_l_add(arg1::Ptr{cholmod_sparse}, arg2::Ptr{cholmod_sparse},
                                    arg3::Ptr{Cdouble}, arg4::Ptr{Cdouble}, arg5::Cint,
                                    arg6::Cint,
                                    arg7::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_sparse_xtype(to_xdtype, A, Common)
    @ccall libcholmod.cholmod_sparse_xtype(to_xdtype::Cint, A::Ptr{cholmod_sparse},
                                           Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_sparse_xtype(arg1, arg2, arg3)
    @ccall libcholmod.cholmod_l_sparse_xtype(arg1::Cint, arg2::Ptr{cholmod_sparse},
                                             arg3::Ptr{cholmod_common})::Cint
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
    @ccall libcholmod.cholmod_allocate_factor(n::Csize_t,
                                              Common::Ptr{cholmod_common})::Ptr{cholmod_factor}
end

function cholmod_l_allocate_factor(arg1, arg2)
    @ccall libcholmod.cholmod_l_allocate_factor(arg1::Csize_t,
                                                arg2::Ptr{cholmod_common})::Ptr{cholmod_factor}
end

function cholmod_alloc_factor(n, dtype, Common)
    @ccall libcholmod.cholmod_alloc_factor(n::Csize_t, dtype::Cint,
                                           Common::Ptr{cholmod_common})::Ptr{cholmod_factor}
end

function cholmod_l_alloc_factor(arg1, arg2, arg3)
    @ccall libcholmod.cholmod_l_alloc_factor(arg1::Csize_t, arg2::Cint,
                                             arg3::Ptr{cholmod_common})::Ptr{cholmod_factor}
end

function cholmod_free_factor(L, Common)
    @ccall libcholmod.cholmod_free_factor(L::Ptr{Ptr{cholmod_factor}},
                                          Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_free_factor(arg1, arg2)
    @ccall libcholmod.cholmod_l_free_factor(arg1::Ptr{Ptr{cholmod_factor}},
                                            arg2::Ptr{cholmod_common})::Cint
end

function cholmod_reallocate_factor(nznew, L, Common)
    @ccall libcholmod.cholmod_reallocate_factor(nznew::Csize_t, L::Ptr{cholmod_factor},
                                                Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_reallocate_factor(arg1, arg2, arg3)
    @ccall libcholmod.cholmod_l_reallocate_factor(arg1::Csize_t, arg2::Ptr{cholmod_factor},
                                                  arg3::Ptr{cholmod_common})::Cint
end

function cholmod_change_factor(to_xtype, to_ll, to_super, to_packed, to_monotonic, L,
                               Common)
    @ccall libcholmod.cholmod_change_factor(to_xtype::Cint, to_ll::Cint, to_super::Cint,
                                            to_packed::Cint, to_monotonic::Cint,
                                            L::Ptr{cholmod_factor},
                                            Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_change_factor(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    @ccall libcholmod.cholmod_l_change_factor(arg1::Cint, arg2::Cint, arg3::Cint,
                                              arg4::Cint, arg5::Cint,
                                              arg6::Ptr{cholmod_factor},
                                              arg7::Ptr{cholmod_common})::Cint
end

function cholmod_pack_factor(L, Common)
    @ccall libcholmod.cholmod_pack_factor(L::Ptr{cholmod_factor},
                                          Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_pack_factor(arg1, arg2)
    @ccall libcholmod.cholmod_l_pack_factor(arg1::Ptr{cholmod_factor},
                                            arg2::Ptr{cholmod_common})::Cint
end

function cholmod_reallocate_column(j, need, L, Common)
    @ccall libcholmod.cholmod_reallocate_column(j::Csize_t, need::Csize_t,
                                                L::Ptr{cholmod_factor},
                                                Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_reallocate_column(arg1, arg2, arg3, arg4)
    @ccall libcholmod.cholmod_l_reallocate_column(arg1::Csize_t, arg2::Csize_t,
                                                  arg3::Ptr{cholmod_factor},
                                                  arg4::Ptr{cholmod_common})::Cint
end

function cholmod_factor_to_sparse(L, Common)
    @ccall libcholmod.cholmod_factor_to_sparse(L::Ptr{cholmod_factor},
                                               Common::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_l_factor_to_sparse(arg1, arg2)
    @ccall libcholmod.cholmod_l_factor_to_sparse(arg1::Ptr{cholmod_factor},
                                                 arg2::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_copy_factor(L, Common)
    @ccall libcholmod.cholmod_copy_factor(L::Ptr{cholmod_factor},
                                          Common::Ptr{cholmod_common})::Ptr{cholmod_factor}
end

function cholmod_l_copy_factor(arg1, arg2)
    @ccall libcholmod.cholmod_l_copy_factor(arg1::Ptr{cholmod_factor},
                                            arg2::Ptr{cholmod_common})::Ptr{cholmod_factor}
end

function cholmod_factor_xtype(to_xdtype, L, Common)
    @ccall libcholmod.cholmod_factor_xtype(to_xdtype::Cint, L::Ptr{cholmod_factor},
                                           Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_factor_xtype(arg1, arg2, arg3)
    @ccall libcholmod.cholmod_l_factor_xtype(arg1::Cint, arg2::Ptr{cholmod_factor},
                                             arg3::Ptr{cholmod_common})::Cint
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

function cholmod_allocate_dense(nrow, ncol, d, xdtype, Common)
    @ccall libcholmod.cholmod_allocate_dense(nrow::Csize_t, ncol::Csize_t, d::Csize_t,
                                             xdtype::Cint,
                                             Common::Ptr{cholmod_common})::Ptr{cholmod_dense}
end

function cholmod_l_allocate_dense(arg1, arg2, arg3, arg4, arg5)
    @ccall libcholmod.cholmod_l_allocate_dense(arg1::Csize_t, arg2::Csize_t, arg3::Csize_t,
                                               arg4::Cint,
                                               arg5::Ptr{cholmod_common})::Ptr{cholmod_dense}
end

function cholmod_zeros(nrow, ncol, xdtype, Common)
    @ccall libcholmod.cholmod_zeros(nrow::Csize_t, ncol::Csize_t, xdtype::Cint,
                                    Common::Ptr{cholmod_common})::Ptr{cholmod_dense}
end

function cholmod_l_zeros(arg1, arg2, arg3, arg4)
    @ccall libcholmod.cholmod_l_zeros(arg1::Csize_t, arg2::Csize_t, arg3::Cint,
                                      arg4::Ptr{cholmod_common})::Ptr{cholmod_dense}
end

function cholmod_ones(nrow, ncol, xdtype, Common)
    @ccall libcholmod.cholmod_ones(nrow::Csize_t, ncol::Csize_t, xdtype::Cint,
                                   Common::Ptr{cholmod_common})::Ptr{cholmod_dense}
end

function cholmod_l_ones(arg1, arg2, arg3, arg4)
    @ccall libcholmod.cholmod_l_ones(arg1::Csize_t, arg2::Csize_t, arg3::Cint,
                                     arg4::Ptr{cholmod_common})::Ptr{cholmod_dense}
end

function cholmod_eye(nrow, ncol, xdtype, Common)
    @ccall libcholmod.cholmod_eye(nrow::Csize_t, ncol::Csize_t, xdtype::Cint,
                                  Common::Ptr{cholmod_common})::Ptr{cholmod_dense}
end

function cholmod_l_eye(arg1, arg2, arg3, arg4)
    @ccall libcholmod.cholmod_l_eye(arg1::Csize_t, arg2::Csize_t, arg3::Cint,
                                    arg4::Ptr{cholmod_common})::Ptr{cholmod_dense}
end

function cholmod_free_dense(X, Common)
    @ccall libcholmod.cholmod_free_dense(X::Ptr{Ptr{cholmod_dense}},
                                         Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_free_dense(arg1, arg2)
    @ccall libcholmod.cholmod_l_free_dense(arg1::Ptr{Ptr{cholmod_dense}},
                                           arg2::Ptr{cholmod_common})::Cint
end

function cholmod_ensure_dense(X, nrow, ncol, d, xdtype, Common)
    @ccall libcholmod.cholmod_ensure_dense(X::Ptr{Ptr{cholmod_dense}}, nrow::Csize_t,
                                           ncol::Csize_t, d::Csize_t, xdtype::Cint,
                                           Common::Ptr{cholmod_common})::Ptr{cholmod_dense}
end

function cholmod_l_ensure_dense(arg1, arg2, arg3, arg4, arg5, arg6)
    @ccall libcholmod.cholmod_l_ensure_dense(arg1::Ptr{Ptr{cholmod_dense}}, arg2::Csize_t,
                                             arg3::Csize_t, arg4::Csize_t, arg5::Cint,
                                             arg6::Ptr{cholmod_common})::Ptr{cholmod_dense}
end

function cholmod_sparse_to_dense(A, Common)
    @ccall libcholmod.cholmod_sparse_to_dense(A::Ptr{cholmod_sparse},
                                              Common::Ptr{cholmod_common})::Ptr{cholmod_dense}
end

function cholmod_l_sparse_to_dense(arg1, arg2)
    @ccall libcholmod.cholmod_l_sparse_to_dense(arg1::Ptr{cholmod_sparse},
                                                arg2::Ptr{cholmod_common})::Ptr{cholmod_dense}
end

function cholmod_dense_nnz(X, Common)
    @ccall libcholmod.cholmod_dense_nnz(X::Ptr{cholmod_dense},
                                        Common::Ptr{cholmod_common})::Int64
end

function cholmod_l_dense_nnz(arg1, arg2)
    @ccall libcholmod.cholmod_l_dense_nnz(arg1::Ptr{cholmod_dense},
                                          arg2::Ptr{cholmod_common})::Int64
end

function cholmod_dense_to_sparse(X, mode, Common)
    @ccall libcholmod.cholmod_dense_to_sparse(X::Ptr{cholmod_dense}, mode::Cint,
                                              Common::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_l_dense_to_sparse(arg1, arg2, arg3)
    @ccall libcholmod.cholmod_l_dense_to_sparse(arg1::Ptr{cholmod_dense}, arg2::Cint,
                                                arg3::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_copy_dense(X, Common)
    @ccall libcholmod.cholmod_copy_dense(X::Ptr{cholmod_dense},
                                         Common::Ptr{cholmod_common})::Ptr{cholmod_dense}
end

function cholmod_l_copy_dense(arg1, arg2)
    @ccall libcholmod.cholmod_l_copy_dense(arg1::Ptr{cholmod_dense},
                                           arg2::Ptr{cholmod_common})::Ptr{cholmod_dense}
end

function cholmod_copy_dense2(X, Y, Common)
    @ccall libcholmod.cholmod_copy_dense2(X::Ptr{cholmod_dense}, Y::Ptr{cholmod_dense},
                                          Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_copy_dense2(arg1, arg2, arg3)
    @ccall libcholmod.cholmod_l_copy_dense2(arg1::Ptr{cholmod_dense},
                                            arg2::Ptr{cholmod_dense},
                                            arg3::Ptr{cholmod_common})::Cint
end

function cholmod_dense_xtype(to_xdtype, X, Common)
    @ccall libcholmod.cholmod_dense_xtype(to_xdtype::Cint, X::Ptr{cholmod_dense},
                                          Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_dense_xtype(arg1, arg2, arg3)
    @ccall libcholmod.cholmod_l_dense_xtype(arg1::Cint, arg2::Ptr{cholmod_dense},
                                            arg3::Ptr{cholmod_common})::Cint
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

function cholmod_allocate_triplet(nrow, ncol, nzmax, stype, xdtype, Common)
    @ccall libcholmod.cholmod_allocate_triplet(nrow::Csize_t, ncol::Csize_t, nzmax::Csize_t,
                                               stype::Cint, xdtype::Cint,
                                               Common::Ptr{cholmod_common})::Ptr{cholmod_triplet}
end

function cholmod_l_allocate_triplet(arg1, arg2, arg3, arg4, arg5, arg6)
    @ccall libcholmod.cholmod_l_allocate_triplet(arg1::Csize_t, arg2::Csize_t,
                                                 arg3::Csize_t, arg4::Cint, arg5::Cint,
                                                 arg6::Ptr{cholmod_common})::Ptr{cholmod_triplet}
end

function cholmod_free_triplet(T, Common)
    @ccall libcholmod.cholmod_free_triplet(T::Ptr{Ptr{cholmod_triplet}},
                                           Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_free_triplet(arg1, arg2)
    @ccall libcholmod.cholmod_l_free_triplet(arg1::Ptr{Ptr{cholmod_triplet}},
                                             arg2::Ptr{cholmod_common})::Cint
end

function cholmod_reallocate_triplet(nznew, T, Common)
    @ccall libcholmod.cholmod_reallocate_triplet(nznew::Csize_t, T::Ptr{cholmod_triplet},
                                                 Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_reallocate_triplet(arg1, arg2, arg3)
    @ccall libcholmod.cholmod_l_reallocate_triplet(arg1::Csize_t,
                                                   arg2::Ptr{cholmod_triplet},
                                                   arg3::Ptr{cholmod_common})::Cint
end

function cholmod_sparse_to_triplet(A, Common)
    @ccall libcholmod.cholmod_sparse_to_triplet(A::Ptr{cholmod_sparse},
                                                Common::Ptr{cholmod_common})::Ptr{cholmod_triplet}
end

function cholmod_l_sparse_to_triplet(arg1, arg2)
    @ccall libcholmod.cholmod_l_sparse_to_triplet(arg1::Ptr{cholmod_sparse},
                                                  arg2::Ptr{cholmod_common})::Ptr{cholmod_triplet}
end

function cholmod_triplet_to_sparse(T, nzmax, Common)
    @ccall libcholmod.cholmod_triplet_to_sparse(T::Ptr{cholmod_triplet}, nzmax::Csize_t,
                                                Common::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_l_triplet_to_sparse(arg1, arg2, arg3)
    @ccall libcholmod.cholmod_l_triplet_to_sparse(arg1::Ptr{cholmod_triplet}, arg2::Csize_t,
                                                  arg3::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_copy_triplet(T, Common)
    @ccall libcholmod.cholmod_copy_triplet(T::Ptr{cholmod_triplet},
                                           Common::Ptr{cholmod_common})::Ptr{cholmod_triplet}
end

function cholmod_l_copy_triplet(arg1, arg2)
    @ccall libcholmod.cholmod_l_copy_triplet(arg1::Ptr{cholmod_triplet},
                                             arg2::Ptr{cholmod_common})::Ptr{cholmod_triplet}
end

function cholmod_triplet_xtype(to_xdtype, T, Common)
    @ccall libcholmod.cholmod_triplet_xtype(to_xdtype::Cint, T::Ptr{cholmod_triplet},
                                            Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_triplet_xtype(arg1, arg2, arg3)
    @ccall libcholmod.cholmod_l_triplet_xtype(arg1::Cint, arg2::Ptr{cholmod_triplet},
                                              arg3::Ptr{cholmod_common})::Cint
end

function cholmod_malloc(n, size, Common)
    @ccall libcholmod.cholmod_malloc(n::Csize_t, size::Csize_t,
                                     Common::Ptr{cholmod_common})::Ptr{Cvoid}
end

function cholmod_l_malloc(arg1, arg2, arg3)
    @ccall libcholmod.cholmod_l_malloc(arg1::Csize_t, arg2::Csize_t,
                                       arg3::Ptr{cholmod_common})::Ptr{Cvoid}
end

function cholmod_calloc(n, size, Common)
    @ccall libcholmod.cholmod_calloc(n::Csize_t, size::Csize_t,
                                     Common::Ptr{cholmod_common})::Ptr{Cvoid}
end

function cholmod_l_calloc(arg1, arg2, arg3)
    @ccall libcholmod.cholmod_l_calloc(arg1::Csize_t, arg2::Csize_t,
                                       arg3::Ptr{cholmod_common})::Ptr{Cvoid}
end

function cholmod_free(n, size, p, Common)
    @ccall libcholmod.cholmod_free(n::Csize_t, size::Csize_t, p::Ptr{Cvoid},
                                   Common::Ptr{cholmod_common})::Ptr{Cvoid}
end

function cholmod_l_free(arg1, arg2, arg3, arg4)
    @ccall libcholmod.cholmod_l_free(arg1::Csize_t, arg2::Csize_t, arg3::Ptr{Cvoid},
                                     arg4::Ptr{cholmod_common})::Ptr{Cvoid}
end

function cholmod_realloc(nnew, size, p, n, Common)
    @ccall libcholmod.cholmod_realloc(nnew::Csize_t, size::Csize_t, p::Ptr{Cvoid},
                                      n::Ptr{Csize_t},
                                      Common::Ptr{cholmod_common})::Ptr{Cvoid}
end

function cholmod_l_realloc(arg1, arg2, arg3, arg4, arg5)
    @ccall libcholmod.cholmod_l_realloc(arg1::Csize_t, arg2::Csize_t, arg3::Ptr{Cvoid},
                                        arg4::Ptr{Csize_t},
                                        arg5::Ptr{cholmod_common})::Ptr{Cvoid}
end

function cholmod_realloc_multiple(nnew, nint, xdtype, I_block, J_block, X_block, Z_block, n,
                                  Common)
    @ccall libcholmod.cholmod_realloc_multiple(nnew::Csize_t, nint::Cint, xdtype::Cint,
                                               I_block::Ptr{Ptr{Cvoid}},
                                               J_block::Ptr{Ptr{Cvoid}},
                                               X_block::Ptr{Ptr{Cvoid}},
                                               Z_block::Ptr{Ptr{Cvoid}}, n::Ptr{Csize_t},
                                               Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_realloc_multiple(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)
    @ccall libcholmod.cholmod_l_realloc_multiple(arg1::Csize_t, arg2::Cint, arg3::Cint,
                                                 arg4::Ptr{Ptr{Cvoid}},
                                                 arg5::Ptr{Ptr{Cvoid}},
                                                 arg6::Ptr{Ptr{Cvoid}},
                                                 arg7::Ptr{Ptr{Cvoid}}, arg8::Ptr{Csize_t},
                                                 arg9::Ptr{cholmod_common})::Cint
end

function cholmod_check_common(Common)
    @ccall libcholmod.cholmod_check_common(Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_check_common(arg1)
    @ccall libcholmod.cholmod_l_check_common(arg1::Ptr{cholmod_common})::Cint
end

function cholmod_print_common(name, Common)
    @ccall libcholmod.cholmod_print_common(name::Ptr{Cchar},
                                           Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_print_common(arg1, arg2)
    @ccall libcholmod.cholmod_l_print_common(arg1::Ptr{Cchar},
                                             arg2::Ptr{cholmod_common})::Cint
end

function cholmod_gpu_stats(arg1)
    @ccall libcholmod.cholmod_gpu_stats(arg1::Ptr{cholmod_common})::Cint
end

function cholmod_l_gpu_stats(arg1)
    @ccall libcholmod.cholmod_l_gpu_stats(arg1::Ptr{cholmod_common})::Cint
end

function cholmod_check_sparse(A, Common)
    @ccall libcholmod.cholmod_check_sparse(A::Ptr{cholmod_sparse},
                                           Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_check_sparse(arg1, arg2)
    @ccall libcholmod.cholmod_l_check_sparse(arg1::Ptr{cholmod_sparse},
                                             arg2::Ptr{cholmod_common})::Cint
end

function cholmod_print_sparse(A, name, Common)
    @ccall libcholmod.cholmod_print_sparse(A::Ptr{cholmod_sparse}, name::Ptr{Cchar},
                                           Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_print_sparse(arg1, arg2, arg3)
    @ccall libcholmod.cholmod_l_print_sparse(arg1::Ptr{cholmod_sparse}, arg2::Ptr{Cchar},
                                             arg3::Ptr{cholmod_common})::Cint
end

function cholmod_check_dense(X, Common)
    @ccall libcholmod.cholmod_check_dense(X::Ptr{cholmod_dense},
                                          Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_check_dense(arg1, arg2)
    @ccall libcholmod.cholmod_l_check_dense(arg1::Ptr{cholmod_dense},
                                            arg2::Ptr{cholmod_common})::Cint
end

function cholmod_print_dense(X, name, Common)
    @ccall libcholmod.cholmod_print_dense(X::Ptr{cholmod_dense}, name::Ptr{Cchar},
                                          Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_print_dense(arg1, arg2, arg3)
    @ccall libcholmod.cholmod_l_print_dense(arg1::Ptr{cholmod_dense}, arg2::Ptr{Cchar},
                                            arg3::Ptr{cholmod_common})::Cint
end

function cholmod_check_factor(L, Common)
    @ccall libcholmod.cholmod_check_factor(L::Ptr{cholmod_factor},
                                           Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_check_factor(arg1, arg2)
    @ccall libcholmod.cholmod_l_check_factor(arg1::Ptr{cholmod_factor},
                                             arg2::Ptr{cholmod_common})::Cint
end

function cholmod_print_factor(L, name, Common)
    @ccall libcholmod.cholmod_print_factor(L::Ptr{cholmod_factor}, name::Ptr{Cchar},
                                           Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_print_factor(arg1, arg2, arg3)
    @ccall libcholmod.cholmod_l_print_factor(arg1::Ptr{cholmod_factor}, arg2::Ptr{Cchar},
                                             arg3::Ptr{cholmod_common})::Cint
end

function cholmod_check_triplet(T, Common)
    @ccall libcholmod.cholmod_check_triplet(T::Ptr{cholmod_triplet},
                                            Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_check_triplet(arg1, arg2)
    @ccall libcholmod.cholmod_l_check_triplet(arg1::Ptr{cholmod_triplet},
                                              arg2::Ptr{cholmod_common})::Cint
end

function cholmod_print_triplet(T, name, Common)
    @ccall libcholmod.cholmod_print_triplet(T::Ptr{cholmod_triplet}, name::Ptr{Cchar},
                                            Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_print_triplet(arg1, arg2, arg3)
    @ccall libcholmod.cholmod_l_print_triplet(arg1::Ptr{cholmod_triplet}, arg2::Ptr{Cchar},
                                              arg3::Ptr{cholmod_common})::Cint
end

function cholmod_check_subset(Set, len, n, Common)
    @ccall libcholmod.cholmod_check_subset(Set::Ptr{Int32}, len::Int64, n::Csize_t,
                                           Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_check_subset(arg1, arg2, arg3, arg4)
    @ccall libcholmod.cholmod_l_check_subset(arg1::Ptr{Int64}, arg2::Int64, arg3::Csize_t,
                                             arg4::Ptr{cholmod_common})::Cint
end

function cholmod_print_subset(Set, len, n, name, Common)
    @ccall libcholmod.cholmod_print_subset(Set::Ptr{Int32}, len::Int64, n::Csize_t,
                                           name::Ptr{Cchar},
                                           Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_print_subset(arg1, arg2, arg3, arg4, arg5)
    @ccall libcholmod.cholmod_l_print_subset(arg1::Ptr{Int64}, arg2::Int64, arg3::Csize_t,
                                             arg4::Ptr{Cchar},
                                             arg5::Ptr{cholmod_common})::Cint
end

function cholmod_check_perm(Perm, len, n, Common)
    @ccall libcholmod.cholmod_check_perm(Perm::Ptr{Int32}, len::Csize_t, n::Csize_t,
                                         Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_check_perm(arg1, arg2, arg3, arg4)
    @ccall libcholmod.cholmod_l_check_perm(arg1::Ptr{Int64}, arg2::Csize_t, arg3::Csize_t,
                                           arg4::Ptr{cholmod_common})::Cint
end

function cholmod_print_perm(Perm, len, n, name, Common)
    @ccall libcholmod.cholmod_print_perm(Perm::Ptr{Int32}, len::Csize_t, n::Csize_t,
                                         name::Ptr{Cchar},
                                         Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_print_perm(arg1, arg2, arg3, arg4, arg5)
    @ccall libcholmod.cholmod_l_print_perm(arg1::Ptr{Int64}, arg2::Csize_t, arg3::Csize_t,
                                           arg4::Ptr{Cchar},
                                           arg5::Ptr{cholmod_common})::Cint
end

function cholmod_check_parent(Parent, n, Common)
    @ccall libcholmod.cholmod_check_parent(Parent::Ptr{Int32}, n::Csize_t,
                                           Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_check_parent(arg1, arg2, arg3)
    @ccall libcholmod.cholmod_l_check_parent(arg1::Ptr{Int64}, arg2::Csize_t,
                                             arg3::Ptr{cholmod_common})::Cint
end

function cholmod_print_parent(Parent, n, name, Common)
    @ccall libcholmod.cholmod_print_parent(Parent::Ptr{Int32}, n::Csize_t, name::Ptr{Cchar},
                                           Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_print_parent(arg1, arg2, arg3, arg4)
    @ccall libcholmod.cholmod_l_print_parent(arg1::Ptr{Int64}, arg2::Csize_t,
                                             arg3::Ptr{Cchar},
                                             arg4::Ptr{cholmod_common})::Cint
end

function cholmod_read_sparse(f, Common)
    @ccall libcholmod.cholmod_read_sparse(f::Ptr{Libc.FILE},
                                          Common::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_l_read_sparse(arg1, arg2)
    @ccall libcholmod.cholmod_l_read_sparse(arg1::Ptr{Libc.FILE},
                                            arg2::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_read_sparse2(f, dtype, Common)
    @ccall libcholmod.cholmod_read_sparse2(f::Ptr{Libc.FILE}, dtype::Cint,
                                           Common::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_l_read_sparse2(arg1, arg2, arg3)
    @ccall libcholmod.cholmod_l_read_sparse2(arg1::Ptr{Libc.FILE}, arg2::Cint,
                                             arg3::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_read_triplet(f, Common)
    @ccall libcholmod.cholmod_read_triplet(f::Ptr{Libc.FILE},
                                           Common::Ptr{cholmod_common})::Ptr{cholmod_triplet}
end

function cholmod_l_read_triplet(arg1, arg2)
    @ccall libcholmod.cholmod_l_read_triplet(arg1::Ptr{Libc.FILE},
                                             arg2::Ptr{cholmod_common})::Ptr{cholmod_triplet}
end

function cholmod_read_triplet2(f, dtype, Common)
    @ccall libcholmod.cholmod_read_triplet2(f::Ptr{Libc.FILE}, dtype::Cint,
                                            Common::Ptr{cholmod_common})::Ptr{cholmod_triplet}
end

function cholmod_l_read_triplet2(arg1, arg2, arg3)
    @ccall libcholmod.cholmod_l_read_triplet2(arg1::Ptr{Libc.FILE}, arg2::Cint,
                                              arg3::Ptr{cholmod_common})::Ptr{cholmod_triplet}
end

function cholmod_read_dense(f, Common)
    @ccall libcholmod.cholmod_read_dense(f::Ptr{Libc.FILE},
                                         Common::Ptr{cholmod_common})::Ptr{cholmod_dense}
end

function cholmod_l_read_dense(arg1, arg2)
    @ccall libcholmod.cholmod_l_read_dense(arg1::Ptr{Libc.FILE},
                                           arg2::Ptr{cholmod_common})::Ptr{cholmod_dense}
end

function cholmod_read_dense2(f, dtype, Common)
    @ccall libcholmod.cholmod_read_dense2(f::Ptr{Libc.FILE}, dtype::Cint,
                                          Common::Ptr{cholmod_common})::Ptr{cholmod_dense}
end

function cholmod_l_read_dense2(arg1, arg2, arg3)
    @ccall libcholmod.cholmod_l_read_dense2(arg1::Ptr{Libc.FILE}, arg2::Cint,
                                            arg3::Ptr{cholmod_common})::Ptr{cholmod_dense}
end

function cholmod_read_matrix(f, prefer, mtype, Common)
    @ccall libcholmod.cholmod_read_matrix(f::Ptr{Libc.FILE}, prefer::Cint, mtype::Ptr{Cint},
                                          Common::Ptr{cholmod_common})::Ptr{Cvoid}
end

function cholmod_l_read_matrix(arg1, arg2, arg3, arg4)
    @ccall libcholmod.cholmod_l_read_matrix(arg1::Ptr{Libc.FILE}, arg2::Cint,
                                            arg3::Ptr{Cint},
                                            arg4::Ptr{cholmod_common})::Ptr{Cvoid}
end

function cholmod_read_matrix2(f, prefer, dtype, mtype, Common)
    @ccall libcholmod.cholmod_read_matrix2(f::Ptr{Libc.FILE}, prefer::Cint, dtype::Cint,
                                           mtype::Ptr{Cint},
                                           Common::Ptr{cholmod_common})::Ptr{Cvoid}
end

function cholmod_l_read_matrix2(arg1, arg2, arg3, arg4, arg5)
    @ccall libcholmod.cholmod_l_read_matrix2(arg1::Ptr{Libc.FILE}, arg2::Cint, arg3::Cint,
                                             arg4::Ptr{Cint},
                                             arg5::Ptr{cholmod_common})::Ptr{Cvoid}
end

function cholmod_write_sparse(f, A, Z, comments, Common)
    @ccall libcholmod.cholmod_write_sparse(f::Ptr{Libc.FILE}, A::Ptr{cholmod_sparse},
                                           Z::Ptr{cholmod_sparse}, comments::Ptr{Cchar},
                                           Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_write_sparse(arg1, arg2, arg3, c, arg5)
    @ccall libcholmod.cholmod_l_write_sparse(arg1::Ptr{Libc.FILE},
                                             arg2::Ptr{cholmod_sparse},
                                             arg3::Ptr{cholmod_sparse}, c::Ptr{Cchar},
                                             arg5::Ptr{cholmod_common})::Cint
end

function cholmod_write_dense(f, X, comments, Common)
    @ccall libcholmod.cholmod_write_dense(f::Ptr{Libc.FILE}, X::Ptr{cholmod_dense},
                                          comments::Ptr{Cchar},
                                          Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_write_dense(arg1, arg2, arg3, arg4)
    @ccall libcholmod.cholmod_l_write_dense(arg1::Ptr{Libc.FILE}, arg2::Ptr{cholmod_dense},
                                            arg3::Ptr{Cchar},
                                            arg4::Ptr{cholmod_common})::Cint
end

function cholmod_analyze(A, Common)
    @ccall libcholmod.cholmod_analyze(A::Ptr{cholmod_sparse},
                                      Common::Ptr{cholmod_common})::Ptr{cholmod_factor}
end

function cholmod_l_analyze(arg1, arg2)
    @ccall libcholmod.cholmod_l_analyze(arg1::Ptr{cholmod_sparse},
                                        arg2::Ptr{cholmod_common})::Ptr{cholmod_factor}
end

function cholmod_analyze_p(A, UserPerm, fset, fsize, Common)
    @ccall libcholmod.cholmod_analyze_p(A::Ptr{cholmod_sparse}, UserPerm::Ptr{Int32},
                                        fset::Ptr{Int32}, fsize::Csize_t,
                                        Common::Ptr{cholmod_common})::Ptr{cholmod_factor}
end

function cholmod_l_analyze_p(arg1, arg2, arg3, arg4, arg5)
    @ccall libcholmod.cholmod_l_analyze_p(arg1::Ptr{cholmod_sparse}, arg2::Ptr{Int64},
                                          arg3::Ptr{Int64}, arg4::Csize_t,
                                          arg5::Ptr{cholmod_common})::Ptr{cholmod_factor}
end

function cholmod_analyze_p2(for_whom, A, UserPerm, fset, fsize, Common)
    @ccall libcholmod.cholmod_analyze_p2(for_whom::Cint, A::Ptr{cholmod_sparse},
                                         UserPerm::Ptr{Int32}, fset::Ptr{Int32},
                                         fsize::Csize_t,
                                         Common::Ptr{cholmod_common})::Ptr{cholmod_factor}
end

function cholmod_l_analyze_p2(arg1, arg2, arg3, arg4, arg5, arg6)
    @ccall libcholmod.cholmod_l_analyze_p2(arg1::Cint, arg2::Ptr{cholmod_sparse},
                                           arg3::Ptr{Int64}, arg4::Ptr{Int64},
                                           arg5::Csize_t,
                                           arg6::Ptr{cholmod_common})::Ptr{cholmod_factor}
end

function cholmod_factorize(A, L, Common)
    @ccall libcholmod.cholmod_factorize(A::Ptr{cholmod_sparse}, L::Ptr{cholmod_factor},
                                        Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_factorize(arg1, arg2, arg3)
    @ccall libcholmod.cholmod_l_factorize(arg1::Ptr{cholmod_sparse},
                                          arg2::Ptr{cholmod_factor},
                                          arg3::Ptr{cholmod_common})::Cint
end

function cholmod_factorize_p(A, beta, fset, fsize, L, Common)
    @ccall libcholmod.cholmod_factorize_p(A::Ptr{cholmod_sparse}, beta::Ptr{Cdouble},
                                          fset::Ptr{Int32}, fsize::Csize_t,
                                          L::Ptr{cholmod_factor},
                                          Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_factorize_p(arg1, arg2, arg3, arg4, arg5, arg6)
    @ccall libcholmod.cholmod_l_factorize_p(arg1::Ptr{cholmod_sparse}, arg2::Ptr{Cdouble},
                                            arg3::Ptr{Int64}, arg4::Csize_t,
                                            arg5::Ptr{cholmod_factor},
                                            arg6::Ptr{cholmod_common})::Cint
end

function cholmod_solve(sys, L, B, Common)
    @ccall libcholmod.cholmod_solve(sys::Cint, L::Ptr{cholmod_factor},
                                    B::Ptr{cholmod_dense},
                                    Common::Ptr{cholmod_common})::Ptr{cholmod_dense}
end

function cholmod_l_solve(arg1, arg2, arg3, arg4)
    @ccall libcholmod.cholmod_l_solve(arg1::Cint, arg2::Ptr{cholmod_factor},
                                      arg3::Ptr{cholmod_dense},
                                      arg4::Ptr{cholmod_common})::Ptr{cholmod_dense}
end

function cholmod_solve2(sys, L, B, Bset, X_Handle, Xset_Handle, Y_Handle, E_Handle, Common)
    @ccall libcholmod.cholmod_solve2(sys::Cint, L::Ptr{cholmod_factor},
                                     B::Ptr{cholmod_dense}, Bset::Ptr{cholmod_sparse},
                                     X_Handle::Ptr{Ptr{cholmod_dense}},
                                     Xset_Handle::Ptr{Ptr{cholmod_sparse}},
                                     Y_Handle::Ptr{Ptr{cholmod_dense}},
                                     E_Handle::Ptr{Ptr{cholmod_dense}},
                                     Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_solve2(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)
    @ccall libcholmod.cholmod_l_solve2(arg1::Cint, arg2::Ptr{cholmod_factor},
                                       arg3::Ptr{cholmod_dense}, arg4::Ptr{cholmod_sparse},
                                       arg5::Ptr{Ptr{cholmod_dense}},
                                       arg6::Ptr{Ptr{cholmod_sparse}},
                                       arg7::Ptr{Ptr{cholmod_dense}},
                                       arg8::Ptr{Ptr{cholmod_dense}},
                                       arg9::Ptr{cholmod_common})::Cint
end

function cholmod_spsolve(sys, L, B, Common)
    @ccall libcholmod.cholmod_spsolve(sys::Cint, L::Ptr{cholmod_factor},
                                      B::Ptr{cholmod_sparse},
                                      Common::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_l_spsolve(arg1, arg2, arg3, arg4)
    @ccall libcholmod.cholmod_l_spsolve(arg1::Cint, arg2::Ptr{cholmod_factor},
                                        arg3::Ptr{cholmod_sparse},
                                        arg4::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_etree(A, Parent, Common)
    @ccall libcholmod.cholmod_etree(A::Ptr{cholmod_sparse}, Parent::Ptr{Int32},
                                    Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_etree(arg1, arg2, arg3)
    @ccall libcholmod.cholmod_l_etree(arg1::Ptr{cholmod_sparse}, arg2::Ptr{Int64},
                                      arg3::Ptr{cholmod_common})::Cint
end

function cholmod_rowcolcounts(A, fset, fsize, Parent, Post, RowCount, ColCount, First,
                              Level, Common)
    @ccall libcholmod.cholmod_rowcolcounts(A::Ptr{cholmod_sparse}, fset::Ptr{Int32},
                                           fsize::Csize_t, Parent::Ptr{Int32},
                                           Post::Ptr{Int32}, RowCount::Ptr{Int32},
                                           ColCount::Ptr{Int32}, First::Ptr{Int32},
                                           Level::Ptr{Int32},
                                           Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_rowcolcounts(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)
    @ccall libcholmod.cholmod_l_rowcolcounts(arg1::Ptr{cholmod_sparse}, arg2::Ptr{Int64},
                                             arg3::Csize_t, arg4::Ptr{Int64},
                                             arg5::Ptr{Int64}, arg6::Ptr{Int64},
                                             arg7::Ptr{Int64}, arg8::Ptr{Int64},
                                             arg9::Ptr{Int64},
                                             arg10::Ptr{cholmod_common})::Cint
end

function cholmod_analyze_ordering(A, ordering, Perm, fset, fsize, Parent, Post, ColCount,
                                  First, Level, Common)
    @ccall libcholmod.cholmod_analyze_ordering(A::Ptr{cholmod_sparse}, ordering::Cint,
                                               Perm::Ptr{Int32}, fset::Ptr{Int32},
                                               fsize::Csize_t, Parent::Ptr{Int32},
                                               Post::Ptr{Int32}, ColCount::Ptr{Int32},
                                               First::Ptr{Int32}, Level::Ptr{Int32},
                                               Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_analyze_ordering(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9,
                                    arg10, arg11)
    @ccall libcholmod.cholmod_l_analyze_ordering(arg1::Ptr{cholmod_sparse}, arg2::Cint,
                                                 arg3::Ptr{Int64}, arg4::Ptr{Int64},
                                                 arg5::Csize_t, arg6::Ptr{Int64},
                                                 arg7::Ptr{Int64}, arg8::Ptr{Int64},
                                                 arg9::Ptr{Int64}, arg10::Ptr{Int64},
                                                 arg11::Ptr{cholmod_common})::Cint
end

function cholmod_amd(A, fset, fsize, Perm, Common)
    @ccall libcholmod.cholmod_amd(A::Ptr{cholmod_sparse}, fset::Ptr{Int32}, fsize::Csize_t,
                                  Perm::Ptr{Int32}, Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_amd(arg1, arg2, arg3, arg4, arg5)
    @ccall libcholmod.cholmod_l_amd(arg1::Ptr{cholmod_sparse}, arg2::Ptr{Int64},
                                    arg3::Csize_t, arg4::Ptr{Int64},
                                    arg5::Ptr{cholmod_common})::Cint
end

function cholmod_colamd(A, fset, fsize, postorder, Perm, Common)
    @ccall libcholmod.cholmod_colamd(A::Ptr{cholmod_sparse}, fset::Ptr{Int32},
                                     fsize::Csize_t, postorder::Cint, Perm::Ptr{Int32},
                                     Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_colamd(arg1, arg2, arg3, arg4, arg5, arg6)
    @ccall libcholmod.cholmod_l_colamd(arg1::Ptr{cholmod_sparse}, arg2::Ptr{Int64},
                                       arg3::Csize_t, arg4::Cint, arg5::Ptr{Int64},
                                       arg6::Ptr{cholmod_common})::Cint
end

function cholmod_rowfac(A, F, beta, kstart, kend, L, Common)
    @ccall libcholmod.cholmod_rowfac(A::Ptr{cholmod_sparse}, F::Ptr{cholmod_sparse},
                                     beta::Ptr{Cdouble}, kstart::Csize_t, kend::Csize_t,
                                     L::Ptr{cholmod_factor},
                                     Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_rowfac(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    @ccall libcholmod.cholmod_l_rowfac(arg1::Ptr{cholmod_sparse}, arg2::Ptr{cholmod_sparse},
                                       arg3::Ptr{Cdouble}, arg4::Csize_t, arg5::Csize_t,
                                       arg6::Ptr{cholmod_factor},
                                       arg7::Ptr{cholmod_common})::Cint
end

function cholmod_rowfac_mask(A, F, beta, kstart, kend, mask, RLinkUp, L, Common)
    @ccall libcholmod.cholmod_rowfac_mask(A::Ptr{cholmod_sparse}, F::Ptr{cholmod_sparse},
                                          beta::Ptr{Cdouble}, kstart::Csize_t,
                                          kend::Csize_t, mask::Ptr{Int32},
                                          RLinkUp::Ptr{Int32}, L::Ptr{cholmod_factor},
                                          Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_rowfac_mask(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)
    @ccall libcholmod.cholmod_l_rowfac_mask(arg1::Ptr{cholmod_sparse},
                                            arg2::Ptr{cholmod_sparse}, arg3::Ptr{Cdouble},
                                            arg4::Csize_t, arg5::Csize_t, arg6::Ptr{Int64},
                                            arg7::Ptr{Int64}, arg8::Ptr{cholmod_factor},
                                            arg9::Ptr{cholmod_common})::Cint
end

function cholmod_rowfac_mask2(A, F, beta, kstart, kend, mask, maskmark, RLinkUp, L, Common)
    @ccall libcholmod.cholmod_rowfac_mask2(A::Ptr{cholmod_sparse}, F::Ptr{cholmod_sparse},
                                           beta::Ptr{Cdouble}, kstart::Csize_t,
                                           kend::Csize_t, mask::Ptr{Int32}, maskmark::Int32,
                                           RLinkUp::Ptr{Int32}, L::Ptr{cholmod_factor},
                                           Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_rowfac_mask2(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)
    @ccall libcholmod.cholmod_l_rowfac_mask2(arg1::Ptr{cholmod_sparse},
                                             arg2::Ptr{cholmod_sparse}, arg3::Ptr{Cdouble},
                                             arg4::Csize_t, arg5::Csize_t, arg6::Ptr{Int64},
                                             arg7::Int64, arg8::Ptr{Int64},
                                             arg9::Ptr{cholmod_factor},
                                             arg10::Ptr{cholmod_common})::Cint
end

function cholmod_row_subtree(A, F, krow, Parent, R, Common)
    @ccall libcholmod.cholmod_row_subtree(A::Ptr{cholmod_sparse}, F::Ptr{cholmod_sparse},
                                          krow::Csize_t, Parent::Ptr{Int32},
                                          R::Ptr{cholmod_sparse},
                                          Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_row_subtree(arg1, arg2, arg3, arg4, arg5, arg6)
    @ccall libcholmod.cholmod_l_row_subtree(arg1::Ptr{cholmod_sparse},
                                            arg2::Ptr{cholmod_sparse}, arg3::Csize_t,
                                            arg4::Ptr{Int64}, arg5::Ptr{cholmod_sparse},
                                            arg6::Ptr{cholmod_common})::Cint
end

function cholmod_lsolve_pattern(B, L, Yset, Common)
    @ccall libcholmod.cholmod_lsolve_pattern(B::Ptr{cholmod_sparse}, L::Ptr{cholmod_factor},
                                             Yset::Ptr{cholmod_sparse},
                                             Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_lsolve_pattern(arg1, arg2, arg3, arg4)
    @ccall libcholmod.cholmod_l_lsolve_pattern(arg1::Ptr{cholmod_sparse},
                                               arg2::Ptr{cholmod_factor},
                                               arg3::Ptr{cholmod_sparse},
                                               arg4::Ptr{cholmod_common})::Cint
end

function cholmod_row_lsubtree(A, Fi, fnz, krow, L, R, Common)
    @ccall libcholmod.cholmod_row_lsubtree(A::Ptr{cholmod_sparse}, Fi::Ptr{Int32},
                                           fnz::Csize_t, krow::Csize_t,
                                           L::Ptr{cholmod_factor}, R::Ptr{cholmod_sparse},
                                           Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_row_lsubtree(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    @ccall libcholmod.cholmod_l_row_lsubtree(arg1::Ptr{cholmod_sparse}, arg2::Ptr{Int64},
                                             arg3::Csize_t, arg4::Csize_t,
                                             arg5::Ptr{cholmod_factor},
                                             arg6::Ptr{cholmod_sparse},
                                             arg7::Ptr{cholmod_common})::Cint
end

function cholmod_resymbol(A, fset, fsize, pack, L, Common)
    @ccall libcholmod.cholmod_resymbol(A::Ptr{cholmod_sparse}, fset::Ptr{Cint},
                                       fsize::Csize_t, pack::Cint, L::Ptr{cholmod_factor},
                                       Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_resymbol(arg1, arg2, arg3, arg4, arg5, arg6)
    @ccall libcholmod.cholmod_l_resymbol(arg1::Ptr{cholmod_sparse}, arg2::Ptr{Int64},
                                         arg3::Csize_t, arg4::Cint,
                                         arg5::Ptr{cholmod_factor},
                                         arg6::Ptr{cholmod_common})::Cint
end

function cholmod_resymbol_noperm(A, fset, fsize, pack, L, Common)
    @ccall libcholmod.cholmod_resymbol_noperm(A::Ptr{cholmod_sparse}, fset::Ptr{Int32},
                                              fsize::Csize_t, pack::Cint,
                                              L::Ptr{cholmod_factor},
                                              Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_resymbol_noperm(arg1, arg2, arg3, arg4, arg5, arg6)
    @ccall libcholmod.cholmod_l_resymbol_noperm(arg1::Ptr{cholmod_sparse}, arg2::Ptr{Int64},
                                                arg3::Csize_t, arg4::Cint,
                                                arg5::Ptr{cholmod_factor},
                                                arg6::Ptr{cholmod_common})::Cint
end

function cholmod_rcond(L, Common)
    @ccall libcholmod.cholmod_rcond(L::Ptr{cholmod_factor},
                                    Common::Ptr{cholmod_common})::Cdouble
end

function cholmod_l_rcond(arg1, arg2)
    @ccall libcholmod.cholmod_l_rcond(arg1::Ptr{cholmod_factor},
                                      arg2::Ptr{cholmod_common})::Cdouble
end

function cholmod_postorder(Parent, n, Weight, Post, Common)
    @ccall libcholmod.cholmod_postorder(Parent::Ptr{Int32}, n::Csize_t, Weight::Ptr{Int32},
                                        Post::Ptr{Int32},
                                        Common::Ptr{cholmod_common})::Int32
end

function cholmod_l_postorder(arg1, arg2, arg3, arg4, arg5)
    @ccall libcholmod.cholmod_l_postorder(arg1::Ptr{Int64}, arg2::Csize_t, arg3::Ptr{Int64},
                                          arg4::Ptr{Int64},
                                          arg5::Ptr{cholmod_common})::Int64
end

function cholmod_drop(tol, A, Common)
    @ccall libcholmod.cholmod_drop(tol::Cdouble, A::Ptr{cholmod_sparse},
                                   Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_drop(arg1, arg2, arg3)
    @ccall libcholmod.cholmod_l_drop(arg1::Cdouble, arg2::Ptr{cholmod_sparse},
                                     arg3::Ptr{cholmod_common})::Cint
end

function cholmod_norm_dense(X, norm, Common)
    @ccall libcholmod.cholmod_norm_dense(X::Ptr{cholmod_dense}, norm::Cint,
                                         Common::Ptr{cholmod_common})::Cdouble
end

function cholmod_l_norm_dense(arg1, arg2, arg3)
    @ccall libcholmod.cholmod_l_norm_dense(arg1::Ptr{cholmod_dense}, arg2::Cint,
                                           arg3::Ptr{cholmod_common})::Cdouble
end

function cholmod_norm_sparse(A, norm, Common)
    @ccall libcholmod.cholmod_norm_sparse(A::Ptr{cholmod_sparse}, norm::Cint,
                                          Common::Ptr{cholmod_common})::Cdouble
end

function cholmod_l_norm_sparse(arg1, arg2, arg3)
    @ccall libcholmod.cholmod_l_norm_sparse(arg1::Ptr{cholmod_sparse}, arg2::Cint,
                                            arg3::Ptr{cholmod_common})::Cdouble
end

function cholmod_horzcat(A, B, mode, Common)
    @ccall libcholmod.cholmod_horzcat(A::Ptr{cholmod_sparse}, B::Ptr{cholmod_sparse},
                                      mode::Cint,
                                      Common::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_l_horzcat(arg1, arg2, arg3, arg4)
    @ccall libcholmod.cholmod_l_horzcat(arg1::Ptr{cholmod_sparse},
                                        arg2::Ptr{cholmod_sparse}, arg3::Cint,
                                        arg4::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_scale(S, scale, A, Common)
    @ccall libcholmod.cholmod_scale(S::Ptr{cholmod_dense}, scale::Cint,
                                    A::Ptr{cholmod_sparse},
                                    Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_scale(arg1, arg2, arg3, arg4)
    @ccall libcholmod.cholmod_l_scale(arg1::Ptr{cholmod_dense}, arg2::Cint,
                                      arg3::Ptr{cholmod_sparse},
                                      arg4::Ptr{cholmod_common})::Cint
end

function cholmod_sdmult(A, transpose, alpha, beta, X, Y, Common)
    @ccall libcholmod.cholmod_sdmult(A::Ptr{cholmod_sparse}, transpose::Cint,
                                     alpha::Ptr{Cdouble}, beta::Ptr{Cdouble},
                                     X::Ptr{cholmod_dense}, Y::Ptr{cholmod_dense},
                                     Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_sdmult(arg1, arg2, arg3, arg4, arg5, Y, arg7)
    @ccall libcholmod.cholmod_l_sdmult(arg1::Ptr{cholmod_sparse}, arg2::Cint,
                                       arg3::Ptr{Cdouble}, arg4::Ptr{Cdouble},
                                       arg5::Ptr{cholmod_dense}, Y::Ptr{cholmod_dense},
                                       arg7::Ptr{cholmod_common})::Cint
end

function cholmod_ssmult(A, B, stype, mode, sorted, Common)
    @ccall libcholmod.cholmod_ssmult(A::Ptr{cholmod_sparse}, B::Ptr{cholmod_sparse},
                                     stype::Cint, mode::Cint, sorted::Cint,
                                     Common::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_l_ssmult(arg1, arg2, arg3, arg4, arg5, arg6)
    @ccall libcholmod.cholmod_l_ssmult(arg1::Ptr{cholmod_sparse}, arg2::Ptr{cholmod_sparse},
                                       arg3::Cint, arg4::Cint, arg5::Cint,
                                       arg6::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_submatrix(A, rset, rsize, cset, csize, mode, sorted, Common)
    @ccall libcholmod.cholmod_submatrix(A::Ptr{cholmod_sparse}, rset::Ptr{Int32},
                                        rsize::Int64, cset::Ptr{Int32}, csize::Int64,
                                        mode::Cint, sorted::Cint,
                                        Common::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_l_submatrix(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
    @ccall libcholmod.cholmod_l_submatrix(arg1::Ptr{cholmod_sparse}, arg2::Ptr{Int64},
                                          arg3::Int64, arg4::Ptr{Int64}, arg5::Int64,
                                          arg6::Cint, arg7::Cint,
                                          arg8::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_vertcat(A, B, mode, Common)
    @ccall libcholmod.cholmod_vertcat(A::Ptr{cholmod_sparse}, B::Ptr{cholmod_sparse},
                                      mode::Cint,
                                      Common::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_l_vertcat(arg1, arg2, arg3, arg4)
    @ccall libcholmod.cholmod_l_vertcat(arg1::Ptr{cholmod_sparse},
                                        arg2::Ptr{cholmod_sparse}, arg3::Cint,
                                        arg4::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

function cholmod_symmetry(A, option, xmatched, pmatched, nzoffdiag, nzdiag, Common)
    @ccall libcholmod.cholmod_symmetry(A::Ptr{cholmod_sparse}, option::Cint,
                                       xmatched::Ptr{Int32}, pmatched::Ptr{Int32},
                                       nzoffdiag::Ptr{Int32}, nzdiag::Ptr{Int32},
                                       Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_symmetry(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    @ccall libcholmod.cholmod_l_symmetry(arg1::Ptr{cholmod_sparse}, arg2::Cint,
                                         arg3::Ptr{Int64}, arg4::Ptr{Int64},
                                         arg5::Ptr{Int64}, arg6::Ptr{Int64},
                                         arg7::Ptr{cholmod_common})::Cint
end

function cholmod_updown(update, C, L, Common)
    @ccall libcholmod.cholmod_updown(update::Cint, C::Ptr{cholmod_sparse},
                                     L::Ptr{cholmod_factor},
                                     Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_updown(arg1, arg2, arg3, arg4)
    @ccall libcholmod.cholmod_l_updown(arg1::Cint, arg2::Ptr{cholmod_sparse},
                                       arg3::Ptr{cholmod_factor},
                                       arg4::Ptr{cholmod_common})::Cint
end

function cholmod_updown_solve(update, C, L, X, DeltaB, Common)
    @ccall libcholmod.cholmod_updown_solve(update::Cint, C::Ptr{cholmod_sparse},
                                           L::Ptr{cholmod_factor}, X::Ptr{cholmod_dense},
                                           DeltaB::Ptr{cholmod_dense},
                                           Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_updown_solve(arg1, arg2, arg3, arg4, arg5, arg6)
    @ccall libcholmod.cholmod_l_updown_solve(arg1::Cint, arg2::Ptr{cholmod_sparse},
                                             arg3::Ptr{cholmod_factor},
                                             arg4::Ptr{cholmod_dense},
                                             arg5::Ptr{cholmod_dense},
                                             arg6::Ptr{cholmod_common})::Cint
end

function cholmod_updown_mark(update, C, colmark, L, X, DeltaB, Common)
    @ccall libcholmod.cholmod_updown_mark(update::Cint, C::Ptr{cholmod_sparse},
                                          colmark::Ptr{Int32}, L::Ptr{cholmod_factor},
                                          X::Ptr{cholmod_dense}, DeltaB::Ptr{cholmod_dense},
                                          Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_updown_mark(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    @ccall libcholmod.cholmod_l_updown_mark(arg1::Cint, arg2::Ptr{cholmod_sparse},
                                            arg3::Ptr{Int64}, arg4::Ptr{cholmod_factor},
                                            arg5::Ptr{cholmod_dense},
                                            arg6::Ptr{cholmod_dense},
                                            arg7::Ptr{cholmod_common})::Cint
end

function cholmod_updown_mask(update, C, colmark, mask, L, X, DeltaB, Common)
    @ccall libcholmod.cholmod_updown_mask(update::Cint, C::Ptr{cholmod_sparse},
                                          colmark::Ptr{Int32}, mask::Ptr{Int32},
                                          L::Ptr{cholmod_factor}, X::Ptr{cholmod_dense},
                                          DeltaB::Ptr{cholmod_dense},
                                          Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_updown_mask(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
    @ccall libcholmod.cholmod_l_updown_mask(arg1::Cint, arg2::Ptr{cholmod_sparse},
                                            arg3::Ptr{Int64}, arg4::Ptr{Int64},
                                            arg5::Ptr{cholmod_factor},
                                            arg6::Ptr{cholmod_dense},
                                            arg7::Ptr{cholmod_dense},
                                            arg8::Ptr{cholmod_common})::Cint
end

function cholmod_updown_mask2(update, C, colmark, mask, maskmark, L, X, DeltaB, Common)
    @ccall libcholmod.cholmod_updown_mask2(update::Cint, C::Ptr{cholmod_sparse},
                                           colmark::Ptr{Int32}, mask::Ptr{Int32},
                                           maskmark::Int32, L::Ptr{cholmod_factor},
                                           X::Ptr{cholmod_dense},
                                           DeltaB::Ptr{cholmod_dense},
                                           Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_updown_mask2(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)
    @ccall libcholmod.cholmod_l_updown_mask2(arg1::Cint, arg2::Ptr{cholmod_sparse},
                                             arg3::Ptr{Int64}, arg4::Ptr{Int64},
                                             arg5::Int64, arg6::Ptr{cholmod_factor},
                                             arg7::Ptr{cholmod_dense},
                                             arg8::Ptr{cholmod_dense},
                                             arg9::Ptr{cholmod_common})::Cint
end

function cholmod_rowadd(k, R, L, Common)
    @ccall libcholmod.cholmod_rowadd(k::Csize_t, R::Ptr{cholmod_sparse},
                                     L::Ptr{cholmod_factor},
                                     Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_rowadd(arg1, arg2, arg3, arg4)
    @ccall libcholmod.cholmod_l_rowadd(arg1::Csize_t, arg2::Ptr{cholmod_sparse},
                                       arg3::Ptr{cholmod_factor},
                                       arg4::Ptr{cholmod_common})::Cint
end

function cholmod_rowadd_solve(k, R, bk, L, X, DeltaB, Common)
    @ccall libcholmod.cholmod_rowadd_solve(k::Csize_t, R::Ptr{cholmod_sparse},
                                           bk::Ptr{Cdouble}, L::Ptr{cholmod_factor},
                                           X::Ptr{cholmod_dense},
                                           DeltaB::Ptr{cholmod_dense},
                                           Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_rowadd_solve(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    @ccall libcholmod.cholmod_l_rowadd_solve(arg1::Csize_t, arg2::Ptr{cholmod_sparse},
                                             arg3::Ptr{Cdouble}, arg4::Ptr{cholmod_factor},
                                             arg5::Ptr{cholmod_dense},
                                             arg6::Ptr{cholmod_dense},
                                             arg7::Ptr{cholmod_common})::Cint
end

function cholmod_rowadd_mark(k, R, bk, colmark, L, X, DeltaB, Common)
    @ccall libcholmod.cholmod_rowadd_mark(k::Csize_t, R::Ptr{cholmod_sparse},
                                          bk::Ptr{Cdouble}, colmark::Ptr{Int32},
                                          L::Ptr{cholmod_factor}, X::Ptr{cholmod_dense},
                                          DeltaB::Ptr{cholmod_dense},
                                          Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_rowadd_mark(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
    @ccall libcholmod.cholmod_l_rowadd_mark(arg1::Csize_t, arg2::Ptr{cholmod_sparse},
                                            arg3::Ptr{Cdouble}, arg4::Ptr{Int64},
                                            arg5::Ptr{cholmod_factor},
                                            arg6::Ptr{cholmod_dense},
                                            arg7::Ptr{cholmod_dense},
                                            arg8::Ptr{cholmod_common})::Cint
end

function cholmod_rowdel(k, R, L, Common)
    @ccall libcholmod.cholmod_rowdel(k::Csize_t, R::Ptr{cholmod_sparse},
                                     L::Ptr{cholmod_factor},
                                     Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_rowdel(arg1, arg2, arg3, arg4)
    @ccall libcholmod.cholmod_l_rowdel(arg1::Csize_t, arg2::Ptr{cholmod_sparse},
                                       arg3::Ptr{cholmod_factor},
                                       arg4::Ptr{cholmod_common})::Cint
end

function cholmod_rowdel_solve(k, R, yk, L, X, DeltaB, Common)
    @ccall libcholmod.cholmod_rowdel_solve(k::Csize_t, R::Ptr{cholmod_sparse},
                                           yk::Ptr{Cdouble}, L::Ptr{cholmod_factor},
                                           X::Ptr{cholmod_dense},
                                           DeltaB::Ptr{cholmod_dense},
                                           Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_rowdel_solve(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    @ccall libcholmod.cholmod_l_rowdel_solve(arg1::Csize_t, arg2::Ptr{cholmod_sparse},
                                             arg3::Ptr{Cdouble}, arg4::Ptr{cholmod_factor},
                                             arg5::Ptr{cholmod_dense},
                                             arg6::Ptr{cholmod_dense},
                                             arg7::Ptr{cholmod_common})::Cint
end

function cholmod_rowdel_mark(k, R, yk, colmark, L, X, DeltaB, Common)
    @ccall libcholmod.cholmod_rowdel_mark(k::Csize_t, R::Ptr{cholmod_sparse},
                                          yk::Ptr{Cdouble}, colmark::Ptr{Int32},
                                          L::Ptr{cholmod_factor}, X::Ptr{cholmod_dense},
                                          DeltaB::Ptr{cholmod_dense},
                                          Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_rowdel_mark(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
    @ccall libcholmod.cholmod_l_rowdel_mark(arg1::Csize_t, arg2::Ptr{cholmod_sparse},
                                            arg3::Ptr{Cdouble}, arg4::Ptr{Int64},
                                            arg5::Ptr{cholmod_factor},
                                            arg6::Ptr{cholmod_dense},
                                            arg7::Ptr{cholmod_dense},
                                            arg8::Ptr{cholmod_common})::Cint
end

function cholmod_ccolamd(A, fset, fsize, Cmember, Perm, Common)
    @ccall libcholmod.cholmod_ccolamd(A::Ptr{cholmod_sparse}, fset::Ptr{Int32},
                                      fsize::Csize_t, Cmember::Ptr{Int32}, Perm::Ptr{Int32},
                                      Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_ccolamd(arg1, arg2, arg3, arg4, arg5, arg6)
    @ccall libcholmod.cholmod_l_ccolamd(arg1::Ptr{cholmod_sparse}, arg2::Ptr{Int64},
                                        arg3::Csize_t, arg4::Ptr{Int64}, arg5::Ptr{Int64},
                                        arg6::Ptr{cholmod_common})::Cint
end

function cholmod_csymamd(A, Cmember, Perm, Common)
    @ccall libcholmod.cholmod_csymamd(A::Ptr{cholmod_sparse}, Cmember::Ptr{Int32},
                                      Perm::Ptr{Int32}, Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_csymamd(arg1, arg2, arg3, arg4)
    @ccall libcholmod.cholmod_l_csymamd(arg1::Ptr{cholmod_sparse}, arg2::Ptr{Int64},
                                        arg3::Ptr{Int64}, arg4::Ptr{cholmod_common})::Cint
end

function cholmod_camd(A, fset, fsize, Cmember, Perm, Common)
    @ccall libcholmod.cholmod_camd(A::Ptr{cholmod_sparse}, fset::Ptr{Int32}, fsize::Csize_t,
                                   Cmember::Ptr{Int32}, Perm::Ptr{Int32},
                                   Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_camd(arg1, arg2, arg3, arg4, arg5, arg6)
    @ccall libcholmod.cholmod_l_camd(arg1::Ptr{cholmod_sparse}, arg2::Ptr{Int64},
                                     arg3::Csize_t, arg4::Ptr{Int64}, arg5::Ptr{Int64},
                                     arg6::Ptr{cholmod_common})::Cint
end

function cholmod_nested_dissection(A, fset, fsize, Perm, CParent, Cmember, Common)
    @ccall libcholmod.cholmod_nested_dissection(A::Ptr{cholmod_sparse}, fset::Ptr{Int32},
                                                fsize::Csize_t, Perm::Ptr{Int32},
                                                CParent::Ptr{Int32}, Cmember::Ptr{Int32},
                                                Common::Ptr{cholmod_common})::Int64
end

function cholmod_l_nested_dissection(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    @ccall libcholmod.cholmod_l_nested_dissection(arg1::Ptr{cholmod_sparse},
                                                  arg2::Ptr{Int64}, arg3::Csize_t,
                                                  arg4::Ptr{Int64}, arg5::Ptr{Int64},
                                                  arg6::Ptr{Int64},
                                                  arg7::Ptr{cholmod_common})::Int64
end

function cholmod_metis(A, fset, fsize, postorder, Perm, Common)
    @ccall libcholmod.cholmod_metis(A::Ptr{cholmod_sparse}, fset::Ptr{Int32},
                                    fsize::Csize_t, postorder::Cint, Perm::Ptr{Int32},
                                    Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_metis(arg1, arg2, arg3, arg4, arg5, arg6)
    @ccall libcholmod.cholmod_l_metis(arg1::Ptr{cholmod_sparse}, arg2::Ptr{Int64},
                                      arg3::Csize_t, arg4::Cint, arg5::Ptr{Int64},
                                      arg6::Ptr{cholmod_common})::Cint
end

function cholmod_bisect(A, fset, fsize, compress, Partition, Common)
    @ccall libcholmod.cholmod_bisect(A::Ptr{cholmod_sparse}, fset::Ptr{Int32},
                                     fsize::Csize_t, compress::Cint, Partition::Ptr{Int32},
                                     Common::Ptr{cholmod_common})::Int64
end

function cholmod_l_bisect(arg1, arg2, arg3, arg4, arg5, arg6)
    @ccall libcholmod.cholmod_l_bisect(arg1::Ptr{cholmod_sparse}, arg2::Ptr{Int64},
                                       arg3::Csize_t, arg4::Cint, arg5::Ptr{Int64},
                                       arg6::Ptr{cholmod_common})::Int64
end

function cholmod_metis_bisector(A, Anw, Aew, Partition, Common)
    @ccall libcholmod.cholmod_metis_bisector(A::Ptr{cholmod_sparse}, Anw::Ptr{Int32},
                                             Aew::Ptr{Int32}, Partition::Ptr{Int32},
                                             Common::Ptr{cholmod_common})::Int64
end

function cholmod_l_metis_bisector(arg1, arg2, arg3, arg4, arg5)
    @ccall libcholmod.cholmod_l_metis_bisector(arg1::Ptr{cholmod_sparse}, arg2::Ptr{Int64},
                                               arg3::Ptr{Int64}, arg4::Ptr{Int64},
                                               arg5::Ptr{cholmod_common})::Int64
end

function cholmod_collapse_septree(n, ncomponents, nd_oksep, nd_small, CParent, Cmember,
                                  Common)
    @ccall libcholmod.cholmod_collapse_septree(n::Csize_t, ncomponents::Csize_t,
                                               nd_oksep::Cdouble, nd_small::Csize_t,
                                               CParent::Ptr{Int32}, Cmember::Ptr{Int32},
                                               Common::Ptr{cholmod_common})::Int64
end

function cholmod_l_collapse_septree(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    @ccall libcholmod.cholmod_l_collapse_septree(arg1::Csize_t, arg2::Csize_t,
                                                 arg3::Cdouble, arg4::Csize_t,
                                                 arg5::Ptr{Int64}, arg6::Ptr{Int64},
                                                 arg7::Ptr{cholmod_common})::Int64
end

function cholmod_super_symbolic(A, F, Parent, L, Common)
    @ccall libcholmod.cholmod_super_symbolic(A::Ptr{cholmod_sparse}, F::Ptr{cholmod_sparse},
                                             Parent::Ptr{Int32}, L::Ptr{cholmod_factor},
                                             Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_super_symbolic(arg1, arg2, arg3, arg4, arg5)
    @ccall libcholmod.cholmod_l_super_symbolic(arg1::Ptr{cholmod_sparse},
                                               arg2::Ptr{cholmod_sparse}, arg3::Ptr{Int64},
                                               arg4::Ptr{cholmod_factor},
                                               arg5::Ptr{cholmod_common})::Cint
end

function cholmod_super_symbolic2(for_whom, A, F, Parent, L, Common)
    @ccall libcholmod.cholmod_super_symbolic2(for_whom::Cint, A::Ptr{cholmod_sparse},
                                              F::Ptr{cholmod_sparse}, Parent::Ptr{Int32},
                                              L::Ptr{cholmod_factor},
                                              Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_super_symbolic2(arg1, arg2, arg3, arg4, arg5, arg6)
    @ccall libcholmod.cholmod_l_super_symbolic2(arg1::Cint, arg2::Ptr{cholmod_sparse},
                                                arg3::Ptr{cholmod_sparse}, arg4::Ptr{Int64},
                                                arg5::Ptr{cholmod_factor},
                                                arg6::Ptr{cholmod_common})::Cint
end

function cholmod_super_numeric(A, F, beta, L, Common)
    @ccall libcholmod.cholmod_super_numeric(A::Ptr{cholmod_sparse}, F::Ptr{cholmod_sparse},
                                            beta::Ptr{Cdouble}, L::Ptr{cholmod_factor},
                                            Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_super_numeric(arg1, arg2, arg3, arg4, arg5)
    @ccall libcholmod.cholmod_l_super_numeric(arg1::Ptr{cholmod_sparse},
                                              arg2::Ptr{cholmod_sparse}, arg3::Ptr{Cdouble},
                                              arg4::Ptr{cholmod_factor},
                                              arg5::Ptr{cholmod_common})::Cint
end

function cholmod_super_lsolve(L, X, E, Common)
    @ccall libcholmod.cholmod_super_lsolve(L::Ptr{cholmod_factor}, X::Ptr{cholmod_dense},
                                           E::Ptr{cholmod_dense},
                                           Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_super_lsolve(arg1, arg2, arg3, arg4)
    @ccall libcholmod.cholmod_l_super_lsolve(arg1::Ptr{cholmod_factor},
                                             arg2::Ptr{cholmod_dense},
                                             arg3::Ptr{cholmod_dense},
                                             arg4::Ptr{cholmod_common})::Cint
end

function cholmod_super_ltsolve(L, X, E, Common)
    @ccall libcholmod.cholmod_super_ltsolve(L::Ptr{cholmod_factor}, X::Ptr{cholmod_dense},
                                            E::Ptr{cholmod_dense},
                                            Common::Ptr{cholmod_common})::Cint
end

function cholmod_l_super_ltsolve(arg1, arg2, arg3, arg4)
    @ccall libcholmod.cholmod_l_super_ltsolve(arg1::Ptr{cholmod_factor},
                                              arg2::Ptr{cholmod_dense},
                                              arg3::Ptr{cholmod_dense},
                                              arg4::Ptr{cholmod_common})::Cint
end

mutable struct cholmod_descendant_score_t
    score::Cdouble
    d::Int64
    cholmod_descendant_score_t() = new()
end

const descendantScore = cholmod_descendant_score_t

function cholmod_score_comp(i, j)
    @ccall libcholmod.cholmod_score_comp(i::Ptr{cholmod_descendant_score_t},
                                         j::Ptr{cholmod_descendant_score_t})::Cint
end

function cholmod_l_score_comp(i, j)
    @ccall libcholmod.cholmod_l_score_comp(i::Ptr{cholmod_descendant_score_t},
                                           j::Ptr{cholmod_descendant_score_t})::Cint
end

function SuiteSparseQR_C(ordering, tol, econ, getCTX, A, Bsparse, Bdense, Zsparse, Zdense,
                         R, E, H, HPinv, HTau, cc)
    @ccall libspqr.SuiteSparseQR_C(ordering::Cint, tol::Cdouble, econ::Int64, getCTX::Cint,
                                   A::Ptr{cholmod_sparse}, Bsparse::Ptr{cholmod_sparse},
                                   Bdense::Ptr{cholmod_dense},
                                   Zsparse::Ptr{Ptr{cholmod_sparse}},
                                   Zdense::Ptr{Ptr{cholmod_dense}},
                                   R::Ptr{Ptr{cholmod_sparse}}, E::Ptr{Ptr{Int64}},
                                   H::Ptr{Ptr{cholmod_sparse}}, HPinv::Ptr{Ptr{Int64}},
                                   HTau::Ptr{Ptr{cholmod_dense}},
                                   cc::Ptr{cholmod_common})::Int64
end

function SuiteSparseQR_i_C(ordering, tol, econ, getCTX, A, Bsparse, Bdense, Zsparse, Zdense,
                           R, E, H, HPinv, HTau, cc)
    @ccall libspqr.SuiteSparseQR_i_C(ordering::Cint, tol::Cdouble, econ::Int32,
                                     getCTX::Cint, A::Ptr{cholmod_sparse},
                                     Bsparse::Ptr{cholmod_sparse},
                                     Bdense::Ptr{cholmod_dense},
                                     Zsparse::Ptr{Ptr{cholmod_sparse}},
                                     Zdense::Ptr{Ptr{cholmod_dense}},
                                     R::Ptr{Ptr{cholmod_sparse}}, E::Ptr{Ptr{Int32}},
                                     H::Ptr{Ptr{cholmod_sparse}}, HPinv::Ptr{Ptr{Int32}},
                                     HTau::Ptr{Ptr{cholmod_dense}},
                                     cc::Ptr{cholmod_common})::Int32
end

function SuiteSparseQR_C_QR(ordering, tol, econ, A, Q, R, E, cc)
    @ccall libspqr.SuiteSparseQR_C_QR(ordering::Cint, tol::Cdouble, econ::Int64,
                                      A::Ptr{cholmod_sparse}, Q::Ptr{Ptr{cholmod_sparse}},
                                      R::Ptr{Ptr{cholmod_sparse}}, E::Ptr{Ptr{Int64}},
                                      cc::Ptr{cholmod_common})::Int64
end

function SuiteSparseQR_i_C_QR(ordering, tol, econ, A, Q, R, E, cc)
    @ccall libspqr.SuiteSparseQR_i_C_QR(ordering::Cint, tol::Cdouble, econ::Int32,
                                        A::Ptr{cholmod_sparse}, Q::Ptr{Ptr{cholmod_sparse}},
                                        R::Ptr{Ptr{cholmod_sparse}}, E::Ptr{Ptr{Int32}},
                                        cc::Ptr{cholmod_common})::Int32
end

function SuiteSparseQR_C_backslash(ordering, tol, A, B, cc)
    @ccall libspqr.SuiteSparseQR_C_backslash(ordering::Cint, tol::Cdouble,
                                             A::Ptr{cholmod_sparse}, B::Ptr{cholmod_dense},
                                             cc::Ptr{cholmod_common})::Ptr{cholmod_dense}
end

function SuiteSparseQR_C_backslash_default(A, B, cc)
    @ccall libspqr.SuiteSparseQR_C_backslash_default(A::Ptr{cholmod_sparse},
                                                     B::Ptr{cholmod_dense},
                                                     cc::Ptr{cholmod_common})::Ptr{cholmod_dense}
end

function SuiteSparseQR_C_backslash_sparse(ordering, tol, A, B, cc)
    @ccall libspqr.SuiteSparseQR_C_backslash_sparse(ordering::Cint, tol::Cdouble,
                                                    A::Ptr{cholmod_sparse},
                                                    B::Ptr{cholmod_sparse},
                                                    cc::Ptr{cholmod_common})::Ptr{cholmod_sparse}
end

mutable struct SuiteSparseQR_C_factorization_struct
    xtype::Cint
    itype::Cint
    factors::Ptr{Cvoid}
    SuiteSparseQR_C_factorization_struct() = new()
end

const SuiteSparseQR_C_factorization = SuiteSparseQR_C_factorization_struct

function SuiteSparseQR_C_factorize(ordering, tol, A, cc)
    @ccall libspqr.SuiteSparseQR_C_factorize(ordering::Cint, tol::Cdouble,
                                             A::Ptr{cholmod_sparse},
                                             cc::Ptr{cholmod_common})::Ptr{SuiteSparseQR_C_factorization}
end

function SuiteSparseQR_C_symbolic(ordering, allow_tol, A, cc)
    @ccall libspqr.SuiteSparseQR_C_symbolic(ordering::Cint, allow_tol::Cint,
                                            A::Ptr{cholmod_sparse},
                                            cc::Ptr{cholmod_common})::Ptr{SuiteSparseQR_C_factorization}
end

function SuiteSparseQR_C_numeric(tol, A, QR, cc)
    @ccall libspqr.SuiteSparseQR_C_numeric(tol::Cdouble, A::Ptr{cholmod_sparse},
                                           QR::Ptr{SuiteSparseQR_C_factorization},
                                           cc::Ptr{cholmod_common})::Cint
end

function SuiteSparseQR_C_free(QR, cc)
    @ccall libspqr.SuiteSparseQR_C_free(QR::Ptr{Ptr{SuiteSparseQR_C_factorization}},
                                        cc::Ptr{cholmod_common})::Cint
end

function SuiteSparseQR_C_solve(system, QR, B, cc)
    @ccall libspqr.SuiteSparseQR_C_solve(system::Cint,
                                         QR::Ptr{SuiteSparseQR_C_factorization},
                                         B::Ptr{cholmod_dense},
                                         cc::Ptr{cholmod_common})::Ptr{cholmod_dense}
end

function SuiteSparseQR_C_qmult(method, QR, X, cc)
    @ccall libspqr.SuiteSparseQR_C_qmult(method::Cint,
                                         QR::Ptr{SuiteSparseQR_C_factorization},
                                         X::Ptr{cholmod_dense},
                                         cc::Ptr{cholmod_common})::Ptr{cholmod_dense}
end

function SuiteSparseQR_C_version(version)
    @ccall libspqr.SuiteSparseQR_C_version(version::Ptr{Cint})::Cvoid
end

function amd_order(n, Ap, Ai, P, Control, Info)
    @ccall libamd.amd_order(n::Int32, Ap::Ptr{Int32}, Ai::Ptr{Int32}, P::Ptr{Int32},
                            Control::Ptr{Cdouble}, Info::Ptr{Cdouble})::Cint
end

function amd_l_order(n, Ap, Ai, P, Control, Info)
    @ccall libamd.amd_l_order(n::Int64, Ap::Ptr{Int64}, Ai::Ptr{Int64}, P::Ptr{Int64},
                              Control::Ptr{Cdouble}, Info::Ptr{Cdouble})::Cint
end

function amd_2(n, Pe, Iw, Len, iwlen, pfree, Nv, Next, Last, Head, Elen, Degree, W, Control,
               Info)
    @ccall libamd.amd_2(n::Int32, Pe::Ptr{Int32}, Iw::Ptr{Int32}, Len::Ptr{Int32},
                        iwlen::Int32, pfree::Int32, Nv::Ptr{Int32}, Next::Ptr{Int32},
                        Last::Ptr{Int32}, Head::Ptr{Int32}, Elen::Ptr{Int32},
                        Degree::Ptr{Int32}, W::Ptr{Int32}, Control::Ptr{Cdouble},
                        Info::Ptr{Cdouble})::Cvoid
end

function amd_l2(n, Pe, Iw, Len, iwlen, pfree, Nv, Next, Last, Head, Elen, Degree, W,
                Control, Info)
    @ccall libamd.amd_l2(n::Int64, Pe::Ptr{Int64}, Iw::Ptr{Int64}, Len::Ptr{Int64},
                         iwlen::Int64, pfree::Int64, Nv::Ptr{Int64}, Next::Ptr{Int64},
                         Last::Ptr{Int64}, Head::Ptr{Int64}, Elen::Ptr{Int64},
                         Degree::Ptr{Int64}, W::Ptr{Int64}, Control::Ptr{Cdouble},
                         Info::Ptr{Cdouble})::Cvoid
end

function amd_valid(n_row, n_col, Ap, Ai)
    @ccall libamd.amd_valid(n_row::Int32, n_col::Int32, Ap::Ptr{Int32},
                            Ai::Ptr{Int32})::Cint
end

function amd_l_valid(n_row, n_col, Ap, Ai)
    @ccall libamd.amd_l_valid(n_row::Int64, n_col::Int64, Ap::Ptr{Int64},
                              Ai::Ptr{Int64})::Cint
end

function amd_defaults(Control)
    @ccall libamd.amd_defaults(Control::Ptr{Cdouble})::Cvoid
end

function amd_l_defaults(Control)
    @ccall libamd.amd_l_defaults(Control::Ptr{Cdouble})::Cvoid
end

function amd_control(Control)
    @ccall libamd.amd_control(Control::Ptr{Cdouble})::Cvoid
end

function amd_l_control(Control)
    @ccall libamd.amd_l_control(Control::Ptr{Cdouble})::Cvoid
end

function amd_info(Info)
    @ccall libamd.amd_info(Info::Ptr{Cdouble})::Cvoid
end

function amd_l_info(Info)
    @ccall libamd.amd_l_info(Info::Ptr{Cdouble})::Cvoid
end

function amd_version(version)
    @ccall libamd.amd_version(version::Ptr{Cint})::Cvoid
end

function umfpack_version(version)
    @ccall libumfpack.umfpack_version(version::Ptr{Cint})::Cvoid
end

function umfpack_di_symbolic(n_row, n_col, Ap, Ai, Ax, Symbolic, Control, Info)
    @ccall libumfpack.umfpack_di_symbolic(n_row::Int32, n_col::Int32, Ap::Ptr{Int32},
                                          Ai::Ptr{Int32}, Ax::Ptr{Cdouble},
                                          Symbolic::Ptr{Ptr{Cvoid}}, Control::Ptr{Cdouble},
                                          Info::Ptr{Cdouble})::Cint
end

function umfpack_dl_symbolic(n_row, n_col, Ap, Ai, Ax, Symbolic, Control, Info)
    @ccall libumfpack.umfpack_dl_symbolic(n_row::Int64, n_col::Int64, Ap::Ptr{Int64},
                                          Ai::Ptr{Int64}, Ax::Ptr{Cdouble},
                                          Symbolic::Ptr{Ptr{Cvoid}}, Control::Ptr{Cdouble},
                                          Info::Ptr{Cdouble})::Cint
end

function umfpack_zi_symbolic(n_row, n_col, Ap, Ai, Ax, Az, Symbolic, Control, Info)
    @ccall libumfpack.umfpack_zi_symbolic(n_row::Int32, n_col::Int32, Ap::Ptr{Int32},
                                          Ai::Ptr{Int32}, Ax::Ptr{Cdouble},
                                          Az::Ptr{Cdouble}, Symbolic::Ptr{Ptr{Cvoid}},
                                          Control::Ptr{Cdouble}, Info::Ptr{Cdouble})::Cint
end

function umfpack_zl_symbolic(n_row, n_col, Ap, Ai, Ax, Az, Symbolic, Control, Info)
    @ccall libumfpack.umfpack_zl_symbolic(n_row::Int64, n_col::Int64, Ap::Ptr{Int64},
                                          Ai::Ptr{Int64}, Ax::Ptr{Cdouble},
                                          Az::Ptr{Cdouble}, Symbolic::Ptr{Ptr{Cvoid}},
                                          Control::Ptr{Cdouble}, Info::Ptr{Cdouble})::Cint
end

function umfpack_di_numeric(Ap, Ai, Ax, Symbolic, Numeric, Control, Info)
    @ccall libumfpack.umfpack_di_numeric(Ap::Ptr{Int32}, Ai::Ptr{Int32}, Ax::Ptr{Cdouble},
                                         Symbolic::Ptr{Cvoid}, Numeric::Ptr{Ptr{Cvoid}},
                                         Control::Ptr{Cdouble}, Info::Ptr{Cdouble})::Cint
end

function umfpack_dl_numeric(Ap, Ai, Ax, Symbolic, Numeric, Control, Info)
    @ccall libumfpack.umfpack_dl_numeric(Ap::Ptr{Int64}, Ai::Ptr{Int64}, Ax::Ptr{Cdouble},
                                         Symbolic::Ptr{Cvoid}, Numeric::Ptr{Ptr{Cvoid}},
                                         Control::Ptr{Cdouble}, Info::Ptr{Cdouble})::Cint
end

function umfpack_zi_numeric(Ap, Ai, Ax, Az, Symbolic, Numeric, Control, Info)
    @ccall libumfpack.umfpack_zi_numeric(Ap::Ptr{Int32}, Ai::Ptr{Int32}, Ax::Ptr{Cdouble},
                                         Az::Ptr{Cdouble}, Symbolic::Ptr{Cvoid},
                                         Numeric::Ptr{Ptr{Cvoid}}, Control::Ptr{Cdouble},
                                         Info::Ptr{Cdouble})::Cint
end

function umfpack_zl_numeric(Ap, Ai, Ax, Az, Symbolic, Numeric, Control, Info)
    @ccall libumfpack.umfpack_zl_numeric(Ap::Ptr{Int64}, Ai::Ptr{Int64}, Ax::Ptr{Cdouble},
                                         Az::Ptr{Cdouble}, Symbolic::Ptr{Cvoid},
                                         Numeric::Ptr{Ptr{Cvoid}}, Control::Ptr{Cdouble},
                                         Info::Ptr{Cdouble})::Cint
end

function umfpack_di_solve(sys, Ap, Ai, Ax, X, B, Numeric, Control, Info)
    @ccall libumfpack.umfpack_di_solve(sys::Cint, Ap::Ptr{Int32}, Ai::Ptr{Int32},
                                       Ax::Ptr{Cdouble}, X::Ptr{Cdouble}, B::Ptr{Cdouble},
                                       Numeric::Ptr{Cvoid}, Control::Ptr{Cdouble},
                                       Info::Ptr{Cdouble})::Cint
end

function umfpack_dl_solve(sys, Ap, Ai, Ax, X, B, Numeric, Control, Info)
    @ccall libumfpack.umfpack_dl_solve(sys::Cint, Ap::Ptr{Int64}, Ai::Ptr{Int64},
                                       Ax::Ptr{Cdouble}, X::Ptr{Cdouble}, B::Ptr{Cdouble},
                                       Numeric::Ptr{Cvoid}, Control::Ptr{Cdouble},
                                       Info::Ptr{Cdouble})::Cint
end

function umfpack_zi_solve(sys, Ap, Ai, Ax, Az, Xx, Xz, Bx, Bz, Numeric, Control, Info)
    @ccall libumfpack.umfpack_zi_solve(sys::Cint, Ap::Ptr{Int32}, Ai::Ptr{Int32},
                                       Ax::Ptr{Cdouble}, Az::Ptr{Cdouble}, Xx::Ptr{Cdouble},
                                       Xz::Ptr{Cdouble}, Bx::Ptr{Cdouble}, Bz::Ptr{Cdouble},
                                       Numeric::Ptr{Cvoid}, Control::Ptr{Cdouble},
                                       Info::Ptr{Cdouble})::Cint
end

function umfpack_zl_solve(sys, Ap, Ai, Ax, Az, Xx, Xz, Bx, Bz, Numeric, Control, Info)
    @ccall libumfpack.umfpack_zl_solve(sys::Cint, Ap::Ptr{Int64}, Ai::Ptr{Int64},
                                       Ax::Ptr{Cdouble}, Az::Ptr{Cdouble}, Xx::Ptr{Cdouble},
                                       Xz::Ptr{Cdouble}, Bx::Ptr{Cdouble}, Bz::Ptr{Cdouble},
                                       Numeric::Ptr{Cvoid}, Control::Ptr{Cdouble},
                                       Info::Ptr{Cdouble})::Cint
end

function umfpack_di_free_symbolic(Symbolic)
    @ccall libumfpack.umfpack_di_free_symbolic(Symbolic::Ptr{Ptr{Cvoid}})::Cvoid
end

function umfpack_dl_free_symbolic(Symbolic)
    @ccall libumfpack.umfpack_dl_free_symbolic(Symbolic::Ptr{Ptr{Cvoid}})::Cvoid
end

function umfpack_zi_free_symbolic(Symbolic)
    @ccall libumfpack.umfpack_zi_free_symbolic(Symbolic::Ptr{Ptr{Cvoid}})::Cvoid
end

function umfpack_zl_free_symbolic(Symbolic)
    @ccall libumfpack.umfpack_zl_free_symbolic(Symbolic::Ptr{Ptr{Cvoid}})::Cvoid
end

function umfpack_di_free_numeric(Numeric)
    @ccall libumfpack.umfpack_di_free_numeric(Numeric::Ptr{Ptr{Cvoid}})::Cvoid
end

function umfpack_dl_free_numeric(Numeric)
    @ccall libumfpack.umfpack_dl_free_numeric(Numeric::Ptr{Ptr{Cvoid}})::Cvoid
end

function umfpack_zi_free_numeric(Numeric)
    @ccall libumfpack.umfpack_zi_free_numeric(Numeric::Ptr{Ptr{Cvoid}})::Cvoid
end

function umfpack_zl_free_numeric(Numeric)
    @ccall libumfpack.umfpack_zl_free_numeric(Numeric::Ptr{Ptr{Cvoid}})::Cvoid
end

function umfpack_di_defaults(Control)
    @ccall libumfpack.umfpack_di_defaults(Control::Ptr{Cdouble})::Cvoid
end

function umfpack_dl_defaults(Control)
    @ccall libumfpack.umfpack_dl_defaults(Control::Ptr{Cdouble})::Cvoid
end

function umfpack_zi_defaults(Control)
    @ccall libumfpack.umfpack_zi_defaults(Control::Ptr{Cdouble})::Cvoid
end

function umfpack_zl_defaults(Control)
    @ccall libumfpack.umfpack_zl_defaults(Control::Ptr{Cdouble})::Cvoid
end

function umfpack_di_qsymbolic(n_row, n_col, Ap, Ai, Ax, Qinit, Symbolic, Control, Info)
    @ccall libumfpack.umfpack_di_qsymbolic(n_row::Int32, n_col::Int32, Ap::Ptr{Int32},
                                           Ai::Ptr{Int32}, Ax::Ptr{Cdouble},
                                           Qinit::Ptr{Int32}, Symbolic::Ptr{Ptr{Cvoid}},
                                           Control::Ptr{Cdouble}, Info::Ptr{Cdouble})::Cint
end

function umfpack_dl_qsymbolic(n_row, n_col, Ap, Ai, Ax, Qinit, Symbolic, Control, Info)
    @ccall libumfpack.umfpack_dl_qsymbolic(n_row::Int64, n_col::Int64, Ap::Ptr{Int64},
                                           Ai::Ptr{Int64}, Ax::Ptr{Cdouble},
                                           Qinit::Ptr{Int64}, Symbolic::Ptr{Ptr{Cvoid}},
                                           Control::Ptr{Cdouble}, Info::Ptr{Cdouble})::Cint
end

function umfpack_zi_qsymbolic(n_row, n_col, Ap, Ai, Ax, Az, Qinit, Symbolic, Control, Info)
    @ccall libumfpack.umfpack_zi_qsymbolic(n_row::Int32, n_col::Int32, Ap::Ptr{Int32},
                                           Ai::Ptr{Int32}, Ax::Ptr{Cdouble},
                                           Az::Ptr{Cdouble}, Qinit::Ptr{Int32},
                                           Symbolic::Ptr{Ptr{Cvoid}}, Control::Ptr{Cdouble},
                                           Info::Ptr{Cdouble})::Cint
end

function umfpack_zl_qsymbolic(n_row, n_col, Ap, Ai, Ax, Az, Qinit, Symbolic, Control, Info)
    @ccall libumfpack.umfpack_zl_qsymbolic(n_row::Int64, n_col::Int64, Ap::Ptr{Int64},
                                           Ai::Ptr{Int64}, Ax::Ptr{Cdouble},
                                           Az::Ptr{Cdouble}, Qinit::Ptr{Int64},
                                           Symbolic::Ptr{Ptr{Cvoid}}, Control::Ptr{Cdouble},
                                           Info::Ptr{Cdouble})::Cint
end

function umfpack_di_fsymbolic(n_row, n_col, Ap, Ai, Ax, user_ordering, user_params,
                              Symbolic, Control, Info)
    @ccall libumfpack.umfpack_di_fsymbolic(n_row::Int32, n_col::Int32, Ap::Ptr{Int32},
                                           Ai::Ptr{Int32}, Ax::Ptr{Cdouble},
                                           user_ordering::Ptr{Cvoid},
                                           user_params::Ptr{Cvoid},
                                           Symbolic::Ptr{Ptr{Cvoid}}, Control::Ptr{Cdouble},
                                           Info::Ptr{Cdouble})::Cint
end

function umfpack_dl_fsymbolic(n_row, n_col, Ap, Ai, Ax, user_ordering, user_params,
                              Symbolic, Control, Info)
    @ccall libumfpack.umfpack_dl_fsymbolic(n_row::Int64, n_col::Int64, Ap::Ptr{Int64},
                                           Ai::Ptr{Int64}, Ax::Ptr{Cdouble},
                                           user_ordering::Ptr{Cvoid},
                                           user_params::Ptr{Cvoid},
                                           Symbolic::Ptr{Ptr{Cvoid}}, Control::Ptr{Cdouble},
                                           Info::Ptr{Cdouble})::Cint
end

function umfpack_zi_fsymbolic(n_row, n_col, Ap, Ai, Ax, Az, user_ordering, user_params,
                              Symbolic, Control, Info)
    @ccall libumfpack.umfpack_zi_fsymbolic(n_row::Int32, n_col::Int32, Ap::Ptr{Int32},
                                           Ai::Ptr{Int32}, Ax::Ptr{Cdouble},
                                           Az::Ptr{Cdouble}, user_ordering::Ptr{Cvoid},
                                           user_params::Ptr{Cvoid},
                                           Symbolic::Ptr{Ptr{Cvoid}}, Control::Ptr{Cdouble},
                                           Info::Ptr{Cdouble})::Cint
end

function umfpack_zl_fsymbolic(n_row, n_col, Ap, Ai, Ax, Az, user_ordering, user_params,
                              Symbolic, Control, Info)
    @ccall libumfpack.umfpack_zl_fsymbolic(n_row::Int64, n_col::Int64, Ap::Ptr{Int64},
                                           Ai::Ptr{Int64}, Ax::Ptr{Cdouble},
                                           Az::Ptr{Cdouble}, user_ordering::Ptr{Cvoid},
                                           user_params::Ptr{Cvoid},
                                           Symbolic::Ptr{Ptr{Cvoid}}, Control::Ptr{Cdouble},
                                           Info::Ptr{Cdouble})::Cint
end

function umfpack_di_paru_symbolic(n_row, n_col, Ap, Ai, Ax, Qinit, user_ordering,
                                  user_params, Symbolic, SW, Control, Info)
    @ccall libumfpack.umfpack_di_paru_symbolic(n_row::Int32, n_col::Int32, Ap::Ptr{Int32},
                                               Ai::Ptr{Int32}, Ax::Ptr{Cdouble},
                                               Qinit::Ptr{Int32}, user_ordering::Ptr{Cvoid},
                                               user_params::Ptr{Cvoid},
                                               Symbolic::Ptr{Ptr{Cvoid}},
                                               SW::Ptr{Ptr{Cvoid}}, Control::Ptr{Cdouble},
                                               Info::Ptr{Cdouble})::Cint
end

function umfpack_dl_paru_symbolic(n_row, n_col, Ap, Ai, Ax, Qinit, user_ordering,
                                  user_params, Symbolic, SW, Control, Info)
    @ccall libumfpack.umfpack_dl_paru_symbolic(n_row::Int64, n_col::Int64, Ap::Ptr{Int64},
                                               Ai::Ptr{Int64}, Ax::Ptr{Cdouble},
                                               Qinit::Ptr{Int64}, user_ordering::Ptr{Cvoid},
                                               user_params::Ptr{Cvoid},
                                               Symbolic::Ptr{Ptr{Cvoid}},
                                               SW::Ptr{Ptr{Cvoid}}, Control::Ptr{Cdouble},
                                               Info::Ptr{Cdouble})::Cint
end

function umfpack_zi_paru_symbolic(n_row, n_col, Ap, Ai, Ax, Az, Qinit, user_ordering,
                                  user_params, Symbolic, SW, Control, Info)
    @ccall libumfpack.umfpack_zi_paru_symbolic(n_row::Int32, n_col::Int32, Ap::Ptr{Int32},
                                               Ai::Ptr{Int32}, Ax::Ptr{Cdouble},
                                               Az::Ptr{Cdouble}, Qinit::Ptr{Int32},
                                               user_ordering::Ptr{Cvoid},
                                               user_params::Ptr{Cvoid},
                                               Symbolic::Ptr{Ptr{Cvoid}},
                                               SW::Ptr{Ptr{Cvoid}}, Control::Ptr{Cdouble},
                                               Info::Ptr{Cdouble})::Cint
end

function umfpack_zl_paru_symbolic(n_row, n_col, Ap, Ai, Ax, Az, Qinit, user_ordering,
                                  user_params, Symbolic, SW, Control, Info)
    @ccall libumfpack.umfpack_zl_paru_symbolic(n_row::Int64, n_col::Int64, Ap::Ptr{Int64},
                                               Ai::Ptr{Int64}, Ax::Ptr{Cdouble},
                                               Az::Ptr{Cdouble}, Qinit::Ptr{Int64},
                                               user_ordering::Ptr{Cvoid},
                                               user_params::Ptr{Cvoid},
                                               Symbolic::Ptr{Ptr{Cvoid}},
                                               SW::Ptr{Ptr{Cvoid}}, Control::Ptr{Cdouble},
                                               Info::Ptr{Cdouble})::Cint
end

function umfpack_di_paru_free_sw(SW)
    @ccall libumfpack.umfpack_di_paru_free_sw(SW::Ptr{Ptr{Cvoid}})::Cvoid
end

function umfpack_dl_paru_free_sw(SW)
    @ccall libumfpack.umfpack_dl_paru_free_sw(SW::Ptr{Ptr{Cvoid}})::Cvoid
end

function umfpack_zi_paru_free_sw(SW)
    @ccall libumfpack.umfpack_zi_paru_free_sw(SW::Ptr{Ptr{Cvoid}})::Cvoid
end

function umfpack_zl_paru_free_sw(SW)
    @ccall libumfpack.umfpack_zl_paru_free_sw(SW::Ptr{Ptr{Cvoid}})::Cvoid
end

function umfpack_di_wsolve(sys, Ap, Ai, Ax, X, B, Numeric, Control, Info, Wi, W)
    @ccall libumfpack.umfpack_di_wsolve(sys::Cint, Ap::Ptr{Int32}, Ai::Ptr{Int32},
                                        Ax::Ptr{Cdouble}, X::Ptr{Cdouble}, B::Ptr{Cdouble},
                                        Numeric::Ptr{Cvoid}, Control::Ptr{Cdouble},
                                        Info::Ptr{Cdouble}, Wi::Ptr{Int32},
                                        W::Ptr{Cdouble})::Cint
end

function umfpack_dl_wsolve(sys, Ap, Ai, Ax, X, B, Numeric, Control, Info, Wi, W)
    @ccall libumfpack.umfpack_dl_wsolve(sys::Cint, Ap::Ptr{Int64}, Ai::Ptr{Int64},
                                        Ax::Ptr{Cdouble}, X::Ptr{Cdouble}, B::Ptr{Cdouble},
                                        Numeric::Ptr{Cvoid}, Control::Ptr{Cdouble},
                                        Info::Ptr{Cdouble}, Wi::Ptr{Int64},
                                        W::Ptr{Cdouble})::Cint
end

function umfpack_zi_wsolve(sys, Ap, Ai, Ax, Az, Xx, Xz, Bx, Bz, Numeric, Control, Info, Wi,
                           W)
    @ccall libumfpack.umfpack_zi_wsolve(sys::Int32, Ap::Ptr{Int32}, Ai::Ptr{Int32},
                                        Ax::Ptr{Cdouble}, Az::Ptr{Cdouble},
                                        Xx::Ptr{Cdouble}, Xz::Ptr{Cdouble},
                                        Bx::Ptr{Cdouble}, Bz::Ptr{Cdouble},
                                        Numeric::Ptr{Cvoid}, Control::Ptr{Cdouble},
                                        Info::Ptr{Cdouble}, Wi::Ptr{Int32},
                                        W::Ptr{Cdouble})::Cint
end

function umfpack_zl_wsolve(sys, Ap, Ai, Ax, Az, Xx, Xz, Bx, Bz, Numeric, Control, Info, Wi,
                           W)
    @ccall libumfpack.umfpack_zl_wsolve(sys::Cint, Ap::Ptr{Int64}, Ai::Ptr{Int64},
                                        Ax::Ptr{Cdouble}, Az::Ptr{Cdouble},
                                        Xx::Ptr{Cdouble}, Xz::Ptr{Cdouble},
                                        Bx::Ptr{Cdouble}, Bz::Ptr{Cdouble},
                                        Numeric::Ptr{Cvoid}, Control::Ptr{Cdouble},
                                        Info::Ptr{Cdouble}, Wi::Ptr{Int64},
                                        W::Ptr{Cdouble})::Cint
end

function umfpack_di_triplet_to_col(n_row, n_col, nz, Ti, Tj, Tx, Ap, Ai, Ax, Map)
    @ccall libumfpack.umfpack_di_triplet_to_col(n_row::Int32, n_col::Int32, nz::Int32,
                                                Ti::Ptr{Int32}, Tj::Ptr{Int32},
                                                Tx::Ptr{Cdouble}, Ap::Ptr{Int32},
                                                Ai::Ptr{Int32}, Ax::Ptr{Cdouble},
                                                Map::Ptr{Int32})::Cint
end

function umfpack_dl_triplet_to_col(n_row, n_col, nz, Ti, Tj, Tx, Ap, Ai, Ax, Map)
    @ccall libumfpack.umfpack_dl_triplet_to_col(n_row::Int64, n_col::Int64, nz::Int64,
                                                Ti::Ptr{Int64}, Tj::Ptr{Int64},
                                                Tx::Ptr{Cdouble}, Ap::Ptr{Int64},
                                                Ai::Ptr{Int64}, Ax::Ptr{Cdouble},
                                                Map::Ptr{Int64})::Cint
end

function umfpack_zi_triplet_to_col(n_row, n_col, nz, Ti, Tj, Tx, Tz, Ap, Ai, Ax, Az, Map)
    @ccall libumfpack.umfpack_zi_triplet_to_col(n_row::Int32, n_col::Int32, nz::Int32,
                                                Ti::Ptr{Int32}, Tj::Ptr{Int32},
                                                Tx::Ptr{Cdouble}, Tz::Ptr{Cdouble},
                                                Ap::Ptr{Int32}, Ai::Ptr{Int32},
                                                Ax::Ptr{Cdouble}, Az::Ptr{Cdouble},
                                                Map::Ptr{Int32})::Cint
end

function umfpack_zl_triplet_to_col(n_row, n_col, nz, Ti, Tj, Tx, Tz, Ap, Ai, Ax, Az, Map)
    @ccall libumfpack.umfpack_zl_triplet_to_col(n_row::Int64, n_col::Int64, nz::Int64,
                                                Ti::Ptr{Int64}, Tj::Ptr{Int64},
                                                Tx::Ptr{Cdouble}, Tz::Ptr{Cdouble},
                                                Ap::Ptr{Int64}, Ai::Ptr{Int64},
                                                Ax::Ptr{Cdouble}, Az::Ptr{Cdouble},
                                                Map::Ptr{Int64})::Cint
end

function umfpack_di_col_to_triplet(n_col, Ap, Tj)
    @ccall libumfpack.umfpack_di_col_to_triplet(n_col::Int32, Ap::Ptr{Int32},
                                                Tj::Ptr{Int32})::Cint
end

function umfpack_dl_col_to_triplet(n_col, Ap, Tj)
    @ccall libumfpack.umfpack_dl_col_to_triplet(n_col::Int64, Ap::Ptr{Int64},
                                                Tj::Ptr{Int64})::Cint
end

function umfpack_zi_col_to_triplet(n_col, Ap, Tj)
    @ccall libumfpack.umfpack_zi_col_to_triplet(n_col::Int32, Ap::Ptr{Int32},
                                                Tj::Ptr{Int32})::Cint
end

function umfpack_zl_col_to_triplet(n_col, Ap, Tj)
    @ccall libumfpack.umfpack_zl_col_to_triplet(n_col::Int64, Ap::Ptr{Int64},
                                                Tj::Ptr{Int64})::Cint
end

function umfpack_di_transpose(n_row, n_col, Ap, Ai, Ax, P, Q, Rp, Ri, Rx)
    @ccall libumfpack.umfpack_di_transpose(n_row::Int32, n_col::Int32, Ap::Ptr{Int32},
                                           Ai::Ptr{Int32}, Ax::Ptr{Cdouble}, P::Ptr{Int32},
                                           Q::Ptr{Int32}, Rp::Ptr{Int32}, Ri::Ptr{Int32},
                                           Rx::Ptr{Cdouble})::Cint
end

function umfpack_dl_transpose(n_row, n_col, Ap, Ai, Ax, P, Q, Rp, Ri, Rx)
    @ccall libumfpack.umfpack_dl_transpose(n_row::Int64, n_col::Int64, Ap::Ptr{Int64},
                                           Ai::Ptr{Int64}, Ax::Ptr{Cdouble}, P::Ptr{Int64},
                                           Q::Ptr{Int64}, Rp::Ptr{Int64}, Ri::Ptr{Int64},
                                           Rx::Ptr{Cdouble})::Cint
end

function umfpack_zi_transpose(n_row, n_col, Ap, Ai, Ax, Az, P, Q, Rp, Ri, Rx, Rz,
                              do_conjugate)
    @ccall libumfpack.umfpack_zi_transpose(n_row::Int32, n_col::Int32, Ap::Ptr{Int32},
                                           Ai::Ptr{Int32}, Ax::Ptr{Cdouble},
                                           Az::Ptr{Cdouble}, P::Ptr{Int32}, Q::Ptr{Int32},
                                           Rp::Ptr{Int32}, Ri::Ptr{Int32}, Rx::Ptr{Cdouble},
                                           Rz::Ptr{Cdouble}, do_conjugate::Cint)::Cint
end

function umfpack_zl_transpose(n_row, n_col, Ap, Ai, Ax, Az, P, Q, Rp, Ri, Rx, Rz,
                              do_conjugate)
    @ccall libumfpack.umfpack_zl_transpose(n_row::Int64, n_col::Int64, Ap::Ptr{Int64},
                                           Ai::Ptr{Int64}, Ax::Ptr{Cdouble},
                                           Az::Ptr{Cdouble}, P::Ptr{Int64}, Q::Ptr{Int64},
                                           Rp::Ptr{Int64}, Ri::Ptr{Int64}, Rx::Ptr{Cdouble},
                                           Rz::Ptr{Cdouble}, do_conjugate::Cint)::Cint
end

function umfpack_di_scale(X, B, Numeric)
    @ccall libumfpack.umfpack_di_scale(X::Ptr{Cdouble}, B::Ptr{Cdouble},
                                       Numeric::Ptr{Cvoid})::Cint
end

function umfpack_dl_scale(X, B, Numeric)
    @ccall libumfpack.umfpack_dl_scale(X::Ptr{Cdouble}, B::Ptr{Cdouble},
                                       Numeric::Ptr{Cvoid})::Cint
end

function umfpack_zi_scale(Xx, Xz, Bx, Bz, Numeric)
    @ccall libumfpack.umfpack_zi_scale(Xx::Ptr{Cdouble}, Xz::Ptr{Cdouble}, Bx::Ptr{Cdouble},
                                       Bz::Ptr{Cdouble}, Numeric::Ptr{Cvoid})::Cint
end

function umfpack_zl_scale(Xx, Xz, Bx, Bz, Numeric)
    @ccall libumfpack.umfpack_zl_scale(Xx::Ptr{Cdouble}, Xz::Ptr{Cdouble}, Bx::Ptr{Cdouble},
                                       Bz::Ptr{Cdouble}, Numeric::Ptr{Cvoid})::Cint
end

function umfpack_di_get_lunz(lnz, unz, n_row, n_col, nz_udiag, Numeric)
    @ccall libumfpack.umfpack_di_get_lunz(lnz::Ptr{Int32}, unz::Ptr{Int32},
                                          n_row::Ptr{Int32}, n_col::Ptr{Int32},
                                          nz_udiag::Ptr{Int32}, Numeric::Ptr{Cvoid})::Cint
end

function umfpack_dl_get_lunz(lnz, unz, n_row, n_col, nz_udiag, Numeric)
    @ccall libumfpack.umfpack_dl_get_lunz(lnz::Ptr{Int64}, unz::Ptr{Int64},
                                          n_row::Ptr{Int64}, n_col::Ptr{Int64},
                                          nz_udiag::Ptr{Int64}, Numeric::Ptr{Cvoid})::Cint
end

function umfpack_zi_get_lunz(lnz, unz, n_row, n_col, nz_udiag, Numeric)
    @ccall libumfpack.umfpack_zi_get_lunz(lnz::Ptr{Int32}, unz::Ptr{Int32},
                                          n_row::Ptr{Int32}, n_col::Ptr{Int32},
                                          nz_udiag::Ptr{Int32}, Numeric::Ptr{Cvoid})::Cint
end

function umfpack_zl_get_lunz(lnz, unz, n_row, n_col, nz_udiag, Numeric)
    @ccall libumfpack.umfpack_zl_get_lunz(lnz::Ptr{Int64}, unz::Ptr{Int64},
                                          n_row::Ptr{Int64}, n_col::Ptr{Int64},
                                          nz_udiag::Ptr{Int64}, Numeric::Ptr{Cvoid})::Cint
end

function umfpack_di_get_numeric(Lp, Lj, Lx, Up, Ui, Ux, P, Q, Dx, do_recip, Rs, Numeric)
    @ccall libumfpack.umfpack_di_get_numeric(Lp::Ptr{Int32}, Lj::Ptr{Int32},
                                             Lx::Ptr{Cdouble}, Up::Ptr{Int32},
                                             Ui::Ptr{Int32}, Ux::Ptr{Cdouble},
                                             P::Ptr{Int32}, Q::Ptr{Int32}, Dx::Ptr{Cdouble},
                                             do_recip::Ptr{Int32}, Rs::Ptr{Cdouble},
                                             Numeric::Ptr{Cvoid})::Cint
end

function umfpack_dl_get_numeric(Lp, Lj, Lx, Up, Ui, Ux, P, Q, Dx, do_recip, Rs, Numeric)
    @ccall libumfpack.umfpack_dl_get_numeric(Lp::Ptr{Int64}, Lj::Ptr{Int64},
                                             Lx::Ptr{Cdouble}, Up::Ptr{Int64},
                                             Ui::Ptr{Int64}, Ux::Ptr{Cdouble},
                                             P::Ptr{Int64}, Q::Ptr{Int64}, Dx::Ptr{Cdouble},
                                             do_recip::Ptr{Int64}, Rs::Ptr{Cdouble},
                                             Numeric::Ptr{Cvoid})::Cint
end

function umfpack_zi_get_numeric(Lp, Lj, Lx, Lz, Up, Ui, Ux, Uz, P, Q, Dx, Dz, do_recip, Rs,
                                Numeric)
    @ccall libumfpack.umfpack_zi_get_numeric(Lp::Ptr{Int32}, Lj::Ptr{Int32},
                                             Lx::Ptr{Cdouble}, Lz::Ptr{Cdouble},
                                             Up::Ptr{Int32}, Ui::Ptr{Int32},
                                             Ux::Ptr{Cdouble}, Uz::Ptr{Cdouble},
                                             P::Ptr{Int32}, Q::Ptr{Int32}, Dx::Ptr{Cdouble},
                                             Dz::Ptr{Cdouble}, do_recip::Ptr{Int32},
                                             Rs::Ptr{Cdouble}, Numeric::Ptr{Cvoid})::Cint
end

function umfpack_zl_get_numeric(Lp, Lj, Lx, Lz, Up, Ui, Ux, Uz, P, Q, Dx, Dz, do_recip, Rs,
                                Numeric)
    @ccall libumfpack.umfpack_zl_get_numeric(Lp::Ptr{Int64}, Lj::Ptr{Int64},
                                             Lx::Ptr{Cdouble}, Lz::Ptr{Cdouble},
                                             Up::Ptr{Int64}, Ui::Ptr{Int64},
                                             Ux::Ptr{Cdouble}, Uz::Ptr{Cdouble},
                                             P::Ptr{Int64}, Q::Ptr{Int64}, Dx::Ptr{Cdouble},
                                             Dz::Ptr{Cdouble}, do_recip::Ptr{Int64},
                                             Rs::Ptr{Cdouble}, Numeric::Ptr{Cvoid})::Cint
end

function umfpack_di_get_symbolic(n_row, n_col, n1, nz, nfr, nchains, P, Q, Front_npivcol,
                                 Front_parent, Front_1strow, Front_leftmostdesc,
                                 Chain_start, Chain_maxrows, Chain_maxcols, Dmap, Symbolic)
    @ccall libumfpack.umfpack_di_get_symbolic(n_row::Ptr{Int32}, n_col::Ptr{Int32},
                                              n1::Ptr{Int32}, nz::Ptr{Int32},
                                              nfr::Ptr{Int32}, nchains::Ptr{Int32},
                                              P::Ptr{Int32}, Q::Ptr{Int32},
                                              Front_npivcol::Ptr{Int32},
                                              Front_parent::Ptr{Int32},
                                              Front_1strow::Ptr{Int32},
                                              Front_leftmostdesc::Ptr{Int32},
                                              Chain_start::Ptr{Int32},
                                              Chain_maxrows::Ptr{Int32},
                                              Chain_maxcols::Ptr{Int32}, Dmap::Ptr{Int32},
                                              Symbolic::Ptr{Cvoid})::Cint
end

function umfpack_dl_get_symbolic(n_row, n_col, n1, nz, nfr, nchains, P, Q, Front_npivcol,
                                 Front_parent, Front_1strow, Front_leftmostdesc,
                                 Chain_start, Chain_maxrows, Chain_maxcols, Dmap, Symbolic)
    @ccall libumfpack.umfpack_dl_get_symbolic(n_row::Ptr{Int64}, n_col::Ptr{Int64},
                                              n1::Ptr{Int64}, nz::Ptr{Int64},
                                              nfr::Ptr{Int64}, nchains::Ptr{Int64},
                                              P::Ptr{Int64}, Q::Ptr{Int64},
                                              Front_npivcol::Ptr{Int64},
                                              Front_parent::Ptr{Int64},
                                              Front_1strow::Ptr{Int64},
                                              Front_leftmostdesc::Ptr{Int64},
                                              Chain_start::Ptr{Int64},
                                              Chain_maxrows::Ptr{Int64},
                                              Chain_maxcols::Ptr{Int64}, Dmap::Ptr{Int64},
                                              Symbolic::Ptr{Cvoid})::Cint
end

function umfpack_zi_get_symbolic(n_row, n_col, n1, nz, nfr, nchains, P, Q, Front_npivcol,
                                 Front_parent, Front_1strow, Front_leftmostdesc,
                                 Chain_start, Chain_maxrows, Chain_maxcols, Dmap, Symbolic)
    @ccall libumfpack.umfpack_zi_get_symbolic(n_row::Ptr{Int32}, n_col::Ptr{Int32},
                                              n1::Ptr{Int32}, nz::Ptr{Int32},
                                              nfr::Ptr{Int32}, nchains::Ptr{Int32},
                                              P::Ptr{Int32}, Q::Ptr{Int32},
                                              Front_npivcol::Ptr{Int32},
                                              Front_parent::Ptr{Int32},
                                              Front_1strow::Ptr{Int32},
                                              Front_leftmostdesc::Ptr{Int32},
                                              Chain_start::Ptr{Int32},
                                              Chain_maxrows::Ptr{Int32},
                                              Chain_maxcols::Ptr{Int32}, Dmap::Ptr{Int32},
                                              Symbolic::Ptr{Cvoid})::Cint
end

function umfpack_zl_get_symbolic(n_row, n_col, n1, nz, nfr, nchains, P, Q, Front_npivcol,
                                 Front_parent, Front_1strow, Front_leftmostdesc,
                                 Chain_start, Chain_maxrows, Chain_maxcols, Dmap, Symbolic)
    @ccall libumfpack.umfpack_zl_get_symbolic(n_row::Ptr{Int64}, n_col::Ptr{Int64},
                                              n1::Ptr{Int64}, nz::Ptr{Int64},
                                              nfr::Ptr{Int64}, nchains::Ptr{Int64},
                                              P::Ptr{Int64}, Q::Ptr{Int64},
                                              Front_npivcol::Ptr{Int64},
                                              Front_parent::Ptr{Int64},
                                              Front_1strow::Ptr{Int64},
                                              Front_leftmostdesc::Ptr{Int64},
                                              Chain_start::Ptr{Int64},
                                              Chain_maxrows::Ptr{Int64},
                                              Chain_maxcols::Ptr{Int64}, Dmap::Ptr{Int64},
                                              Symbolic::Ptr{Cvoid})::Cint
end

function umfpack_di_save_numeric(Numeric, filename)
    @ccall libumfpack.umfpack_di_save_numeric(Numeric::Ptr{Cvoid},
                                              filename::Ptr{Cchar})::Cint
end

function umfpack_dl_save_numeric(Numeric, filename)
    @ccall libumfpack.umfpack_dl_save_numeric(Numeric::Ptr{Cvoid},
                                              filename::Ptr{Cchar})::Cint
end

function umfpack_zi_save_numeric(Numeric, filename)
    @ccall libumfpack.umfpack_zi_save_numeric(Numeric::Ptr{Cvoid},
                                              filename::Ptr{Cchar})::Cint
end

function umfpack_zl_save_numeric(Numeric, filename)
    @ccall libumfpack.umfpack_zl_save_numeric(Numeric::Ptr{Cvoid},
                                              filename::Ptr{Cchar})::Cint
end

function umfpack_di_load_numeric(Numeric, filename)
    @ccall libumfpack.umfpack_di_load_numeric(Numeric::Ptr{Ptr{Cvoid}},
                                              filename::Ptr{Cchar})::Cint
end

function umfpack_dl_load_numeric(Numeric, filename)
    @ccall libumfpack.umfpack_dl_load_numeric(Numeric::Ptr{Ptr{Cvoid}},
                                              filename::Ptr{Cchar})::Cint
end

function umfpack_zi_load_numeric(Numeric, filename)
    @ccall libumfpack.umfpack_zi_load_numeric(Numeric::Ptr{Ptr{Cvoid}},
                                              filename::Ptr{Cchar})::Cint
end

function umfpack_zl_load_numeric(Numeric, filename)
    @ccall libumfpack.umfpack_zl_load_numeric(Numeric::Ptr{Ptr{Cvoid}},
                                              filename::Ptr{Cchar})::Cint
end

function umfpack_di_copy_numeric(Numeric, Original)
    @ccall libumfpack.umfpack_di_copy_numeric(Numeric::Ptr{Ptr{Cvoid}},
                                              Original::Ptr{Cvoid})::Cint
end

function umfpack_dl_copy_numeric(Numeric, Original)
    @ccall libumfpack.umfpack_dl_copy_numeric(Numeric::Ptr{Ptr{Cvoid}},
                                              Original::Ptr{Cvoid})::Cint
end

function umfpack_zi_copy_numeric(Numeric, Original)
    @ccall libumfpack.umfpack_zi_copy_numeric(Numeric::Ptr{Ptr{Cvoid}},
                                              Original::Ptr{Cvoid})::Cint
end

function umfpack_zl_copy_numeric(Numeric, Original)
    @ccall libumfpack.umfpack_zl_copy_numeric(Numeric::Ptr{Ptr{Cvoid}},
                                              Original::Ptr{Cvoid})::Cint
end

function umfpack_di_serialize_numeric_size(blobsize, Numeric)
    @ccall libumfpack.umfpack_di_serialize_numeric_size(blobsize::Ptr{Int64},
                                                        Numeric::Ptr{Cvoid})::Cint
end

function umfpack_dl_serialize_numeric_size(blobsize, Numeric)
    @ccall libumfpack.umfpack_dl_serialize_numeric_size(blobsize::Ptr{Int64},
                                                        Numeric::Ptr{Cvoid})::Cint
end

function umfpack_zi_serialize_numeric_size(blobsize, Numeric)
    @ccall libumfpack.umfpack_zi_serialize_numeric_size(blobsize::Ptr{Int64},
                                                        Numeric::Ptr{Cvoid})::Cint
end

function umfpack_zl_serialize_numeric_size(blobsize, Numeric)
    @ccall libumfpack.umfpack_zl_serialize_numeric_size(blobsize::Ptr{Int64},
                                                        Numeric::Ptr{Cvoid})::Cint
end

function umfpack_di_serialize_numeric(blob, blobsize, Numeric)
    @ccall libumfpack.umfpack_di_serialize_numeric(blob::Ptr{Int8}, blobsize::Int64,
                                                   Numeric::Ptr{Cvoid})::Cint
end

function umfpack_dl_serialize_numeric(blob, blobsize, Numeric)
    @ccall libumfpack.umfpack_dl_serialize_numeric(blob::Ptr{Int8}, blobsize::Int64,
                                                   Numeric::Ptr{Cvoid})::Cint
end

function umfpack_zi_serialize_numeric(blob, blobsize, Numeric)
    @ccall libumfpack.umfpack_zi_serialize_numeric(blob::Ptr{Int8}, blobsize::Int64,
                                                   Numeric::Ptr{Cvoid})::Cint
end

function umfpack_zl_serialize_numeric(blob, blobsize, Numeric)
    @ccall libumfpack.umfpack_zl_serialize_numeric(blob::Ptr{Int8}, blobsize::Int64,
                                                   Numeric::Ptr{Cvoid})::Cint
end

function umfpack_di_deserialize_numeric(Numeric, blob, blobsize)
    @ccall libumfpack.umfpack_di_deserialize_numeric(Numeric::Ptr{Ptr{Cvoid}},
                                                     blob::Ptr{Int8}, blobsize::Int64)::Cint
end

function umfpack_dl_deserialize_numeric(Numeric, blob, blobsize)
    @ccall libumfpack.umfpack_dl_deserialize_numeric(Numeric::Ptr{Ptr{Cvoid}},
                                                     blob::Ptr{Int8}, blobsize::Int64)::Cint
end

function umfpack_zi_deserialize_numeric(Numeric, blob, blobsize)
    @ccall libumfpack.umfpack_zi_deserialize_numeric(Numeric::Ptr{Ptr{Cvoid}},
                                                     blob::Ptr{Int8}, blobsize::Int64)::Cint
end

function umfpack_zl_deserialize_numeric(Numeric, blob, blobsize)
    @ccall libumfpack.umfpack_zl_deserialize_numeric(Numeric::Ptr{Ptr{Cvoid}},
                                                     blob::Ptr{Int8}, blobsize::Int64)::Cint
end

function umfpack_di_save_symbolic(Symbolic, filename)
    @ccall libumfpack.umfpack_di_save_symbolic(Symbolic::Ptr{Cvoid},
                                               filename::Ptr{Cchar})::Cint
end

function umfpack_dl_save_symbolic(Symbolic, filename)
    @ccall libumfpack.umfpack_dl_save_symbolic(Symbolic::Ptr{Cvoid},
                                               filename::Ptr{Cchar})::Cint
end

function umfpack_zi_save_symbolic(Symbolic, filename)
    @ccall libumfpack.umfpack_zi_save_symbolic(Symbolic::Ptr{Cvoid},
                                               filename::Ptr{Cchar})::Cint
end

function umfpack_zl_save_symbolic(Symbolic, filename)
    @ccall libumfpack.umfpack_zl_save_symbolic(Symbolic::Ptr{Cvoid},
                                               filename::Ptr{Cchar})::Cint
end

function umfpack_di_load_symbolic(Symbolic, filename)
    @ccall libumfpack.umfpack_di_load_symbolic(Symbolic::Ptr{Ptr{Cvoid}},
                                               filename::Ptr{Cchar})::Cint
end

function umfpack_dl_load_symbolic(Symbolic, filename)
    @ccall libumfpack.umfpack_dl_load_symbolic(Symbolic::Ptr{Ptr{Cvoid}},
                                               filename::Ptr{Cchar})::Cint
end

function umfpack_zi_load_symbolic(Symbolic, filename)
    @ccall libumfpack.umfpack_zi_load_symbolic(Symbolic::Ptr{Ptr{Cvoid}},
                                               filename::Ptr{Cchar})::Cint
end

function umfpack_zl_load_symbolic(Symbolic, filename)
    @ccall libumfpack.umfpack_zl_load_symbolic(Symbolic::Ptr{Ptr{Cvoid}},
                                               filename::Ptr{Cchar})::Cint
end

function umfpack_di_copy_symbolic(Symbolic, Original)
    @ccall libumfpack.umfpack_di_copy_symbolic(Symbolic::Ptr{Ptr{Cvoid}},
                                               Original::Ptr{Cvoid})::Cint
end

function umfpack_dl_copy_symbolic(Symbolic, Original)
    @ccall libumfpack.umfpack_dl_copy_symbolic(Symbolic::Ptr{Ptr{Cvoid}},
                                               Original::Ptr{Cvoid})::Cint
end

function umfpack_zi_copy_symbolic(Symbolic, Original)
    @ccall libumfpack.umfpack_zi_copy_symbolic(Symbolic::Ptr{Ptr{Cvoid}},
                                               Original::Ptr{Cvoid})::Cint
end

function umfpack_zl_copy_symbolic(Symbolic, Original)
    @ccall libumfpack.umfpack_zl_copy_symbolic(Symbolic::Ptr{Ptr{Cvoid}},
                                               Original::Ptr{Cvoid})::Cint
end

function umfpack_di_serialize_symbolic_size(blobsize, Symbolic)
    @ccall libumfpack.umfpack_di_serialize_symbolic_size(blobsize::Ptr{Int64},
                                                         Symbolic::Ptr{Cvoid})::Cint
end

function umfpack_dl_serialize_symbolic_size(blobsize, Symbolic)
    @ccall libumfpack.umfpack_dl_serialize_symbolic_size(blobsize::Ptr{Int64},
                                                         Symbolic::Ptr{Cvoid})::Cint
end

function umfpack_zi_serialize_symbolic_size(blobsize, Symbolic)
    @ccall libumfpack.umfpack_zi_serialize_symbolic_size(blobsize::Ptr{Int64},
                                                         Symbolic::Ptr{Cvoid})::Cint
end

function umfpack_zl_serialize_symbolic_size(blobsize, Symbolic)
    @ccall libumfpack.umfpack_zl_serialize_symbolic_size(blobsize::Ptr{Int64},
                                                         Symbolic::Ptr{Cvoid})::Cint
end

function umfpack_di_serialize_symbolic(blob, blobsize, Symbolic)
    @ccall libumfpack.umfpack_di_serialize_symbolic(blob::Ptr{Int8}, blobsize::Int64,
                                                    Symbolic::Ptr{Cvoid})::Cint
end

function umfpack_dl_serialize_symbolic(blob, blobsize, Symbolic)
    @ccall libumfpack.umfpack_dl_serialize_symbolic(blob::Ptr{Int8}, blobsize::Int64,
                                                    Symbolic::Ptr{Cvoid})::Cint
end

function umfpack_zi_serialize_symbolic(blob, blobsize, Symbolic)
    @ccall libumfpack.umfpack_zi_serialize_symbolic(blob::Ptr{Int8}, blobsize::Int64,
                                                    Symbolic::Ptr{Cvoid})::Cint
end

function umfpack_zl_serialize_symbolic(blob, blobsize, Symbolic)
    @ccall libumfpack.umfpack_zl_serialize_symbolic(blob::Ptr{Int8}, blobsize::Int64,
                                                    Symbolic::Ptr{Cvoid})::Cint
end

function umfpack_di_deserialize_symbolic(Symbolic, blob, blobsize)
    @ccall libumfpack.umfpack_di_deserialize_symbolic(Symbolic::Ptr{Ptr{Cvoid}},
                                                      blob::Ptr{Int8},
                                                      blobsize::Int64)::Cint
end

function umfpack_dl_deserialize_symbolic(Symbolic, blob, blobsize)
    @ccall libumfpack.umfpack_dl_deserialize_symbolic(Symbolic::Ptr{Ptr{Cvoid}},
                                                      blob::Ptr{Int8},
                                                      blobsize::Int64)::Cint
end

function umfpack_zi_deserialize_symbolic(Symbolic, blob, blobsize)
    @ccall libumfpack.umfpack_zi_deserialize_symbolic(Symbolic::Ptr{Ptr{Cvoid}},
                                                      blob::Ptr{Int8},
                                                      blobsize::Int64)::Cint
end

function umfpack_zl_deserialize_symbolic(Symbolic, blob, blobsize)
    @ccall libumfpack.umfpack_zl_deserialize_symbolic(Symbolic::Ptr{Ptr{Cvoid}},
                                                      blob::Ptr{Int8},
                                                      blobsize::Int64)::Cint
end

function umfpack_di_get_determinant(Mx, Ex, Numeric, User_Info)
    @ccall libumfpack.umfpack_di_get_determinant(Mx::Ptr{Cdouble}, Ex::Ptr{Cdouble},
                                                 Numeric::Ptr{Cvoid},
                                                 User_Info::Ptr{Cdouble})::Cint
end

function umfpack_dl_get_determinant(Mx, Ex, Numeric, User_Info)
    @ccall libumfpack.umfpack_dl_get_determinant(Mx::Ptr{Cdouble}, Ex::Ptr{Cdouble},
                                                 Numeric::Ptr{Cvoid},
                                                 User_Info::Ptr{Cdouble})::Cint
end

function umfpack_zi_get_determinant(Mx, Mz, Ex, Numeric, User_Info)
    @ccall libumfpack.umfpack_zi_get_determinant(Mx::Ptr{Cdouble}, Mz::Ptr{Cdouble},
                                                 Ex::Ptr{Cdouble}, Numeric::Ptr{Cvoid},
                                                 User_Info::Ptr{Cdouble})::Cint
end

function umfpack_zl_get_determinant(Mx, Mz, Ex, Numeric, User_Info)
    @ccall libumfpack.umfpack_zl_get_determinant(Mx::Ptr{Cdouble}, Mz::Ptr{Cdouble},
                                                 Ex::Ptr{Cdouble}, Numeric::Ptr{Cvoid},
                                                 User_Info::Ptr{Cdouble})::Cint
end

function umfpack_di_report_status(Control, status)
    @ccall libumfpack.umfpack_di_report_status(Control::Ptr{Cdouble}, status::Cint)::Cvoid
end

function umfpack_dl_report_status(Control, status)
    @ccall libumfpack.umfpack_dl_report_status(Control::Ptr{Cdouble}, status::Cint)::Cvoid
end

function umfpack_zi_report_status(Control, status)
    @ccall libumfpack.umfpack_zi_report_status(Control::Ptr{Cdouble}, status::Cint)::Cvoid
end

function umfpack_zl_report_status(Control, status)
    @ccall libumfpack.umfpack_zl_report_status(Control::Ptr{Cdouble}, status::Cint)::Cvoid
end

function umfpack_di_report_info(Control, Info)
    @ccall libumfpack.umfpack_di_report_info(Control::Ptr{Cdouble},
                                             Info::Ptr{Cdouble})::Cvoid
end

function umfpack_dl_report_info(Control, Info)
    @ccall libumfpack.umfpack_dl_report_info(Control::Ptr{Cdouble},
                                             Info::Ptr{Cdouble})::Cvoid
end

function umfpack_zi_report_info(Control, Info)
    @ccall libumfpack.umfpack_zi_report_info(Control::Ptr{Cdouble},
                                             Info::Ptr{Cdouble})::Cvoid
end

function umfpack_zl_report_info(Control, Info)
    @ccall libumfpack.umfpack_zl_report_info(Control::Ptr{Cdouble},
                                             Info::Ptr{Cdouble})::Cvoid
end

function umfpack_di_report_control(Control)
    @ccall libumfpack.umfpack_di_report_control(Control::Ptr{Cdouble})::Cvoid
end

function umfpack_dl_report_control(Control)
    @ccall libumfpack.umfpack_dl_report_control(Control::Ptr{Cdouble})::Cvoid
end

function umfpack_zi_report_control(Control)
    @ccall libumfpack.umfpack_zi_report_control(Control::Ptr{Cdouble})::Cvoid
end

function umfpack_zl_report_control(Control)
    @ccall libumfpack.umfpack_zl_report_control(Control::Ptr{Cdouble})::Cvoid
end

function umfpack_di_report_matrix(n_row, n_col, Ap, Ai, Ax, col_form, Control)
    @ccall libumfpack.umfpack_di_report_matrix(n_row::Int32, n_col::Int32, Ap::Ptr{Int32},
                                               Ai::Ptr{Int32}, Ax::Ptr{Cdouble},
                                               col_form::Cint, Control::Ptr{Cdouble})::Cint
end

function umfpack_dl_report_matrix(n_row, n_col, Ap, Ai, Ax, col_form, Control)
    @ccall libumfpack.umfpack_dl_report_matrix(n_row::Int64, n_col::Int64, Ap::Ptr{Int64},
                                               Ai::Ptr{Int64}, Ax::Ptr{Cdouble},
                                               col_form::Cint, Control::Ptr{Cdouble})::Cint
end

function umfpack_zi_report_matrix(n_row, n_col, Ap, Ai, Ax, Az, col_form, Control)
    @ccall libumfpack.umfpack_zi_report_matrix(n_row::Int32, n_col::Int32, Ap::Ptr{Int32},
                                               Ai::Ptr{Int32}, Ax::Ptr{Cdouble},
                                               Az::Ptr{Cdouble}, col_form::Cint,
                                               Control::Ptr{Cdouble})::Cint
end

function umfpack_zl_report_matrix(n_row, n_col, Ap, Ai, Ax, Az, col_form, Control)
    @ccall libumfpack.umfpack_zl_report_matrix(n_row::Int64, n_col::Int64, Ap::Ptr{Int64},
                                               Ai::Ptr{Int64}, Ax::Ptr{Cdouble},
                                               Az::Ptr{Cdouble}, col_form::Cint,
                                               Control::Ptr{Cdouble})::Cint
end

function umfpack_di_report_triplet(n_row, n_col, nz, Ti, Tj, Tx, Control)
    @ccall libumfpack.umfpack_di_report_triplet(n_row::Int32, n_col::Int32, nz::Int32,
                                                Ti::Ptr{Int32}, Tj::Ptr{Int32},
                                                Tx::Ptr{Cdouble},
                                                Control::Ptr{Cdouble})::Cint
end

function umfpack_dl_report_triplet(n_row, n_col, nz, Ti, Tj, Tx, Control)
    @ccall libumfpack.umfpack_dl_report_triplet(n_row::Int64, n_col::Int64, nz::Int64,
                                                Ti::Ptr{Int64}, Tj::Ptr{Int64},
                                                Tx::Ptr{Cdouble},
                                                Control::Ptr{Cdouble})::Cint
end

function umfpack_zi_report_triplet(n_row, n_col, nz, Ti, Tj, Tx, Tz, Control)
    @ccall libumfpack.umfpack_zi_report_triplet(n_row::Int32, n_col::Int32, nz::Int32,
                                                Ti::Ptr{Int32}, Tj::Ptr{Int32},
                                                Tx::Ptr{Cdouble}, Tz::Ptr{Cdouble},
                                                Control::Ptr{Cdouble})::Cint
end

function umfpack_zl_report_triplet(n_row, n_col, nz, Ti, Tj, Tx, Tz, Control)
    @ccall libumfpack.umfpack_zl_report_triplet(n_row::Int64, n_col::Int64, nz::Int64,
                                                Ti::Ptr{Int64}, Tj::Ptr{Int64},
                                                Tx::Ptr{Cdouble}, Tz::Ptr{Cdouble},
                                                Control::Ptr{Cdouble})::Cint
end

function umfpack_di_report_vector(n, X, Control)
    @ccall libumfpack.umfpack_di_report_vector(n::Int32, X::Ptr{Cdouble},
                                               Control::Ptr{Cdouble})::Cint
end

function umfpack_dl_report_vector(n, X, Control)
    @ccall libumfpack.umfpack_dl_report_vector(n::Int64, X::Ptr{Cdouble},
                                               Control::Ptr{Cdouble})::Cint
end

function umfpack_zi_report_vector(n, Xx, Xz, Control)
    @ccall libumfpack.umfpack_zi_report_vector(n::Int32, Xx::Ptr{Cdouble}, Xz::Ptr{Cdouble},
                                               Control::Ptr{Cdouble})::Cint
end

function umfpack_zl_report_vector(n, Xx, Xz, Control)
    @ccall libumfpack.umfpack_zl_report_vector(n::Int64, Xx::Ptr{Cdouble}, Xz::Ptr{Cdouble},
                                               Control::Ptr{Cdouble})::Cint
end

function umfpack_di_report_symbolic(Symbolic, Control)
    @ccall libumfpack.umfpack_di_report_symbolic(Symbolic::Ptr{Cvoid},
                                                 Control::Ptr{Cdouble})::Cint
end

function umfpack_dl_report_symbolic(Symbolic, Control)
    @ccall libumfpack.umfpack_dl_report_symbolic(Symbolic::Ptr{Cvoid},
                                                 Control::Ptr{Cdouble})::Cint
end

function umfpack_zi_report_symbolic(Symbolic, Control)
    @ccall libumfpack.umfpack_zi_report_symbolic(Symbolic::Ptr{Cvoid},
                                                 Control::Ptr{Cdouble})::Cint
end

function umfpack_zl_report_symbolic(Symbolic, Control)
    @ccall libumfpack.umfpack_zl_report_symbolic(Symbolic::Ptr{Cvoid},
                                                 Control::Ptr{Cdouble})::Cint
end

function umfpack_di_report_numeric(Numeric, Control)
    @ccall libumfpack.umfpack_di_report_numeric(Numeric::Ptr{Cvoid},
                                                Control::Ptr{Cdouble})::Cint
end

function umfpack_dl_report_numeric(Numeric, Control)
    @ccall libumfpack.umfpack_dl_report_numeric(Numeric::Ptr{Cvoid},
                                                Control::Ptr{Cdouble})::Cint
end

function umfpack_zi_report_numeric(Numeric, Control)
    @ccall libumfpack.umfpack_zi_report_numeric(Numeric::Ptr{Cvoid},
                                                Control::Ptr{Cdouble})::Cint
end

function umfpack_zl_report_numeric(Numeric, Control)
    @ccall libumfpack.umfpack_zl_report_numeric(Numeric::Ptr{Cvoid},
                                                Control::Ptr{Cdouble})::Cint
end

function umfpack_di_report_perm(np, Perm, Control)
    @ccall libumfpack.umfpack_di_report_perm(np::Int32, Perm::Ptr{Int32},
                                             Control::Ptr{Cdouble})::Cint
end

function umfpack_dl_report_perm(np, Perm, Control)
    @ccall libumfpack.umfpack_dl_report_perm(np::Int64, Perm::Ptr{Int64},
                                             Control::Ptr{Cdouble})::Cint
end

function umfpack_zi_report_perm(np, Perm, Control)
    @ccall libumfpack.umfpack_zi_report_perm(np::Int32, Perm::Ptr{Int32},
                                             Control::Ptr{Cdouble})::Cint
end

function umfpack_zl_report_perm(np, Perm, Control)
    @ccall libumfpack.umfpack_zl_report_perm(np::Int64, Perm::Ptr{Int64},
                                             Control::Ptr{Cdouble})::Cint
end

function umfpack_timer()
    @ccall libumfpack.umfpack_timer()::Cdouble
end

function umfpack_tic(stats)
    @ccall libumfpack.umfpack_tic(stats::Ptr{Cdouble})::Cvoid
end

function umfpack_toc(stats)
    @ccall libumfpack.umfpack_toc(stats::Ptr{Cdouble})::Cvoid
end

const CHOLMOD_PATTERN = 0

const CHOLMOD_REAL = 1

const CHOLMOD_COMPLEX = 2

const CHOLMOD_ZOMPLEX = 3

const CHOLMOD_DOUBLE = 0

const CHOLMOD_SINGLE = 4

const CHOLMOD_INT = 0

const CHOLMOD_LONG = 2

const CHOLMOD_DATE = "Feb 20, 2025"

const CHOLMOD_MAIN_VERSION = 5

const CHOLMOD_SUB_VERSION = 3

const CHOLMOD_SUBSUB_VERSION = 1

SUITESPARSE_VER_CODE(main, sub) = main * 1000 + sub

CHOLMOD_VER_CODE(main, sub) = SUITESPARSE_VER_CODE(main, sub)

const CHOLMOD_VERSION = CHOLMOD_VER_CODE(5, 3)

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

const SUITESPARSE_TIME = SuiteSparse_time()

const SUITESPARSE_DATE = "Mar 6, 2025"

const SUITESPARSE_MAIN_VERSION = 7

const SUITESPARSE_SUB_VERSION = 10

const SUITESPARSE_SUBSUB_VERSION = 1

const SUITESPARSE_VERSION = SUITESPARSE_VER_CODE(7, 10)

function SUITESPARSE__VERCODE(main, sub, patch)
    return (main * Culonglong(1000) + sub) * Culonglong(1000) + patch
end

const SUITESPARSE__VERSION = SUITESPARSE__VERCODE(7, 10, 1)

const CHOLMOD__VERSION = SUITESPARSE__VERCODE(5, 3, 1)

const CHOLMOD_DEVICE_SUPERNODE_BUFFERS = 6

const CHOLMOD_HOST_SUPERNODE_BUFFERS = 8

const CHOLMOD_DEVICE_STREAMS = 2

const CHOLMOD_COMMON = 0

const CHOLMOD_SPARSE = 1

const CHOLMOD_FACTOR = 2

const CHOLMOD_DENSE = 3

const CHOLMOD_TRIPLET = 4

const CHOLMOD_SIMPLICIAL = 0

const CHOLMOD_AUTO = 1

const CHOLMOD_SUPERNODAL = 2

const CHOLMOD_NATURAL = 0

const CHOLMOD_GIVEN = 1

const CHOLMOD_AMD = 2

const CHOLMOD_METIS = 3

const CHOLMOD_NESDIS = 4

const CHOLMOD_COLAMD = 5

const CHOLMOD_POSTORDERED = 6

const CHOLMOD_MAXMETHODS = 9

const CHOLMOD_OK = 0

const CHOLMOD_NOT_INSTALLED = -1

const CHOLMOD_OUT_OF_MEMORY = -2

const CHOLMOD_TOO_LARGE = -3

const CHOLMOD_INVALID = -4

const CHOLMOD_GPU_PROBLEM = -5

const CHOLMOD_NOT_POSDEF = 1

const CHOLMOD_DSMALL = 2

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

const CHOLMOD_ANALYZE_FOR_SPQR = 0

const CHOLMOD_ANALYZE_FOR_CHOLESKY = 1

const CHOLMOD_ANALYZE_FOR_SPQRGPU = 2

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

const SPQR_DATE = "June 20, 2024"

const SPQR_MAIN_VERSION = 4

const SPQR_SUB_VERSION = 3

const SPQR_SUBSUB_VERSION = 4

SPQR_VER_CODE(main, sub) = SUITESPARSE_VER_CODE(main, sub)

const SPQR_VERSION = SPQR_VER_CODE(4, 3)

const SPQR__VERSION = SUITESPARSE__VERCODE(4, 3, 4)

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

const AMD_DATE = "June 20, 2024"

const AMD_MAIN_VERSION = 3

const AMD_SUB_VERSION = 3

const AMD_SUBSUB_VERSION = 3

AMD_VERSION_CODE(main, sub) = SUITESPARSE_VER_CODE(main, sub)

const AMD_VERSION = AMD_VERSION_CODE(3, 3)

const AMD__VERSION = SUITESPARSE__VERCODE(3, 3, 3)

const UMFPACK_INFO = 90

const UMFPACK_CONTROL = 20

const UMFPACK_DATE = "Sept 23, 2024"

const UMFPACK_MAIN_VERSION = 6

const UMFPACK_SUB_VERSION = 3

const UMFPACK_SUBSUB_VERSION = 5

UMFPACK_VER_CODE(main, sub) = SUITESPARSE_VER_CODE(main, sub)

const UMFPACK_VER = UMFPACK_VER_CODE(6, 3)

const UMFPACK__VERSION = SUITESPARSE__VERCODE(6, 3, 5)

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
