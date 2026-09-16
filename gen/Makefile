VER=7.14.0
JLL_BUILD=1

all: clean download
	mkdir -p SuiteSparse-$(VER)
	cd SuiteSparse-$(VER) && tar -zxvf ../SuiteSparse.v$(VER).x86_64-linux-gnu.tar.gz
	julia --project -e "using Pkg; Pkg.instantiate()"
	julia --project generator.jl ./SuiteSparse-$(VER)

clean:
	rm -fr *.tar.gz SuiteSparse*

download:
	curl -L -O https://github.com/JuliaBinaryWrappers/SuiteSparse_jll.jl/releases/download/SuiteSparse-v$(VER)%2B$(JLL_BUILD)/SuiteSparse.v$(VER).x86_64-linux-gnu.tar.gz
