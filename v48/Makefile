all: build/v48.pdf

# hier Python-Skripte:
build/int_1.pdf build/int_2.pdf build/T_I_1.pdf build/T_I_2.pdf build/T_I_1_clean.pdf build/T_I_2_clean.pdf build/T_I_1_linfit.pdf build/T_I_2_linfit.pdf build/tau.pdf: plot.py ../matplotlibrc ../header-matplotlib.tex | build
	# so that matplotlib can find the tex header when running
	# LaTeX in the tmp directory
	# and set the matplotlibrc
	TEXINPUTS=$$(pwd)/..: MATPLOTLIBRC=../matplotlibrc python plot.py

# hier weitere Abhängigkeiten für build/v48.pdf deklarieren:
build/v48.pdf: build/int_1.pdf build/int_2.pdf build/T_I_1.pdf build/T_I_2.pdf build/T_I_1_clean.pdf build/T_I_2_clean.pdf build/T_I_1_linfit.pdf build/T_I_2_linfit.pdf build/tau.pdf

build/v48.pdf: FORCE | build
	# to find header and bib files in the main directory
	TEXINPUTS=..: \
	BIBINPUTS=..: \
	max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	v48.tex
	cp build/v48.pdf v48.pdf
#	open "https://www.youtube.com/watch?v=xvFZjo5PgG0"

build:
	mkdir -p build

clean:
	rm -rf build
	rm v48.pdf

FORCE:

.PHONY: all clean
