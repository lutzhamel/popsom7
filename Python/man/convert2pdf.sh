#!/bin/bash
# Convert markdown files to pdf
pandoc -f markdown -t pdf -o sklearnapi.pdf sklearnapi.md
pandoc -f markdown -t pdf -o maputils.pdf maputils.md