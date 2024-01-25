all: build/v51.pdf

# hier Python-Skripte:
build/invert1.pdf build/invert2.pdf build/invert3.pdf build/invert_phase.pdf build/integrator.pdf build/differentiator.pdf build/generator_scope.pdf: plot.py ../matplotlibrc ../header-matplotlib.tex | build
	# so that matplotlib can find the tex header when running
	# LaTeX in the tmp directory
	# and set the matplotlibrc
	TEXINPUTS=$$(pwd)/..: MATPLOTLIBRC=../matplotlibrc python plot.py

# hier weitere Abhängigkeiten für build/v51.pdf deklarieren:
build/v51.pdf: build/invert1.pdf build/invert2.pdf build/invert3.pdf build/invert_phase.pdf build/integrator.pdf build/differentiator.pdf build/generator_scope.pdf

build/v51.pdf: FORCE | build
	# to find header and bib files in the main directory
	TEXINPUTS=..: \
	BIBINPUTS=..: \
	max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	v51.tex
	cp build/v51.pdf v51.pdf
#	open "https://www.youtube.com/watch?v=xvFZjo5PgG0"

build:
	mkdir -p build

clean:
	rm -rf build
	rm v51.pdf

FORCE:

.PHONY: all clean
