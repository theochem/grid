# Generates API html for grid while ignoring the test and data folders.
# Stores it in pyapi/

sphinx-apidoc -a -o pyapi/ ../src/grid ../src/grid/tests/ ../src/grid/test/ ../src/grid/data/ --separate
