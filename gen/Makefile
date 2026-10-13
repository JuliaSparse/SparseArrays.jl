VER=7.14.0
JLL_BUILD=1
SHA256=86e119fd0354cbd22d204f7ce937240cf3c54f0e87834bbb0b01c41245983f12
JULIA ?= julia

SOURCE_DIR=SuiteSparse-$(VER)+$(JLL_BUILD)
ARCHIVE=$(SOURCE_DIR).x86_64-linux-gnu.tar.gz
URL=https://github.com/JuliaBinaryWrappers/SuiteSparse_jll.jl/releases/download/SuiteSparse-v$(VER)%2B$(JLL_BUILD)/SuiteSparse.v$(VER).x86_64-linux-gnu.tar.gz

.PHONY: all clean download
.DELETE_ON_ERROR:

all: $(SOURCE_DIR)/.extracted
	$(JULIA) --project -e "using Pkg; Pkg.instantiate()"
	$(JULIA) --project generator.jl ./$(SOURCE_DIR)

$(SOURCE_DIR)/.extracted: $(ARCHIVE) Makefile
	printf '%s  %s\n' '$(SHA256)' '$(ARCHIVE)' | shasum -a 256 -c -
	rm -rf $(SOURCE_DIR)
	mkdir -p $(SOURCE_DIR)
	tar -zxf $(ARCHIVE) -C $(SOURCE_DIR)
	touch $@

download: $(ARCHIVE)

$(ARCHIVE):
	curl --fail --location --retry 3 --output $@ $(URL)
	printf '%s  %s\n' '$(SHA256)' '$@' | shasum -a 256 -c -

clean:
	rm -rf $(SOURCE_DIR) $(ARCHIVE)
