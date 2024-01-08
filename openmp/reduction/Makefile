.PHONY: test build

config:
	cmake -B ./build -DCMAKE_BUILD_TYPE=Debug

build: config
	cmake --build ./build --target openmp_reduction --config Debug

run: build
	./executable/openmp_reduction

clean:
	rm -rf build/ executable/
