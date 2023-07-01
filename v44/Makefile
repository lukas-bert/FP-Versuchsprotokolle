all: build/v44.pdf

# hier Python-Skripte:
build/Dscan.pdf build/Zscan.pdf build/RockingScan.pdf build/Reflek1.pdf build/Reflek2.pdf build/Reflek3.pdf: plot.py ../matplotlibrc ../header-matplotlib.tex | build
	# so that matplotlib can find the tex header when running
	# LaTeX in the tmp directory
	# and set the matplotlibrc
	TEXINPUTS=$$(pwd)/..: MATPLOTLIBRC=../matplotlibrc python plot.py

# hier weitere Abhängigkeiten für build/v44.pdf deklarieren:
build/v44.pdf: build/Dscan.pdf build/Zscan.pdf build/RockingScan.pdf build/Reflek1.pdf build/Reflek2.pdf build/Reflek3.pdf

build/v44.pdf: FORCE | build
	# to find header and bib files in the main directory
	TEXINPUTS=..: \
	BIBINPUTS=..: \
	max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	v44.tex
	cp build/v44.pdf v44.pdf
#	open "https://www.youtube.com/watch?v=xvFZjo5PgG0"

build:
	mkdir -p build

clean:
	rm -rf build
	rm v44.pdf

FORCE:

.PHONY: all clean
