# Instructions On Documentation

In-order to build the documentation locally, the following commands need to
be runned. The html files are created in an folder called "build".

```bash
cd doc
# # Generates API html for grid while ignoring the test and data folders.
# Stores it in pyapi/
sphinx-apidoc -a -o pyapi/ ../src/grid ../src/grid/tests/ ../src/grid/test/ ../src/grid/data/ --separate
# Build the html files
sphinx-build -M html . ./build
```
Here inside the "./build/html/ folder, there are the html files to run the website locally.


## Pushing To Website

After running the commands above, you'll need to go inside the  "./build/html" folder and copy all the files.

```bash
git branch -a
# Should be something like origin/doc/ where origin is the remote to the theochem/grid Github
git checkout doc
cd ..  # Get out of doc and go back to grid folder
```
Now, paste and overwrite all of the files into ./docs/. Then commit and push to the `doc` branch of the theochem/grid
```bash
git add ./docs/*
git commit -m "Update website"
git push origin doc
```
