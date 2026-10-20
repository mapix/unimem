#!/bin/sh
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode unimem.tex >/dev/null && bibtex -terse unimem && pdflatex -interaction=nonstopmode unimem.tex >/dev/null && pdflatex -interaction=nonstopmode unimem.tex >/dev/null
grep -n "^!\|Overfull\|undefined\|Citation" unimem.log
pdfinfo unimem.pdf | grep Pages
