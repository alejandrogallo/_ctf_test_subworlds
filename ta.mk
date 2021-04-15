TA_REPO = https://github.com/alejandrogallo/tiledarray
TA_PATH = $(abspath ./deps/tildearray)
TA_BUILD = $(TA_PATH)/build
TA_BUILT = $(TA_BUILD)/built
TA_INCLUDE = $(TA_BUILT)/include
TA_LIB = $(TA_BUILT)/lib/libtildearray.a
TA_CMAKE = $(TA_PATH)/CMakeLists.txt

$(TA_CMAKE):
	mkdir -p $(@D)
	git clone $(TA_REPO) $(@D)

$(TA_LIB): $(TA_CMAKE)
	cd $(<D); cmake -B $(TA_BUILD) -D CMAKE_INSTALL_PREFIX=$(TA_BUILT)
	cd $(TA_BUILD); make -j 8
	cd $(TA_BUILD); make install

ta: $(TA_LIB)

.PHONY: ta
