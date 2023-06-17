all: build/v01.pdf

# hier Python-Skripte:
build/plot1.pdf build/plot2.pdf build/fit.pdf build/fit_log.pdf: plot.py ../matplotlibrc ../header-matplotlib.tex | build
	# so that matplotlib can find the tex header when running
	# LaTeX in the tmp directory
	# and set the matplotlibrc
	TEXINPUTS=$$(pwd)/..: MATPLOTLIBRC=../matplotlibrc python plot.py

# hier weitere Abhängigkeiten für build/v01.pdf deklarieren:
build/v01.pdf: build/plot1.pdf build/plot2.pdf build/fit.pdf build/fit_log.pdf
 
build/v01.pdf: FORCE | build
	# to find header and bib files in the main directory
	TEXINPUTS=..: \
	BIBINPUTS=..: \
	max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	v01.tex
	cp build/v01.pdf v01.pdf
#	open "https://www.youtube.com/watch?v=xvFZjo5PgG0"

build:
	mkdir -p build

clean:
	rm -rf build
	rm v01.pdf

FORCE:

.PHONY: all clean
