.PHONY: all test test-concurrent lint dummy

all:
	cargo build

test:
	cargo test
	cargo test --features concurrent --lib --tests --workspace
	cargo test --features concurrent --doc --workspace

lint:
	cargo fmt --all -- --check
	cargo clippy --workspace --all-targets
	cargo check --workspace --no-default-features
	cargo check --workspace --all-targets --all-features

docs:
	RUSTDOCFLAGS="--html-in-header gh-pages/katex-header.html" cargo doc --all --no-deps --document-private-items --all-features
	RUSTDOCFLAGS="--html-in-header gh-pages/katex-header.html" cargo rustdoc --examples --all-features
	# Prevent the examples from showing up in the sidebar
	echo "window.ALL_CRATES = [$$(ls crates/ | sed "s/.*/'&',/; s/-/_/g")'ext'];" > target/doc/crates.js


fix-benchmarks: $(patsubst examples/benchmarks/%, examples/benchmarks/%-fixed, $(wildcard examples/benchmarks/*))
examples/benchmarks/%-fixed: FILE = examples/benchmarks/$*
examples/benchmarks/%-fixed: dummy
	(head -n 1 $(FILE) && bash -c "echo '' | cargo run --example $$(head -n 1 $(FILE))") > $@
	if diff --color $(FILE) $@; then \
	    rm $@; \
	else \
	    mv $@ $(FILE); \
	fi

benchmarks: $(wildcard examples/benchmarks/*)
examples/benchmarks/%: dummy
	(head -n 1 $@ && bash -c "echo '' | cargo run --example $$(head -n 1 $@)") | diff --color $@ -

benchmarks-concurrent: $(patsubst examples/benchmarks/%, examples/benchmarks/%-concurrent, $(wildcard examples/benchmarks/*))

examples/benchmarks/%-concurrent: FILE = examples/benchmarks/$*
examples/benchmarks/%-concurrent: dummy
	(head -n 1 $(FILE) && bash -c "echo '' | cargo run --features concurrent --example $$(head -n 1 $(FILE))") | diff --color $(FILE) -

miri:
	cargo miri test -p once
	MIRIFLAGS="-Zmiri-ignore-leaks -Zmiri-disable-isolation" cargo miri test -p once --features concurrent
