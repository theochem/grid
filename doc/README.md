# Instructions On Documentation

In-order to build the documentation locally, the following commands need to
be runned. The html files are created in an folder called "build".

```bash
# Run one of the following commands to install the required dependencies:
pip install qc-grid[doc]
# Or if you're installing from a local setup:
# pip install .[doc]

cd doc
# # Generates API html for grid while ignoring the test and data folders.
# Stores it in pyapi/
sphinx-apidoc -a -o pyapi/ ../src/grid ../src/grid/tests/ ../src/grid/test/ ../src/grid/data/ --separate
# Build the html files
sphinx-build -M html . ./build
```
Here inside the "./build/html/ folder, there are the html files to run the website locally.


## Pushing To Website

**WARNING: The website is automatically built everytime there is a push to the main branch and thus these sets of instructions must be taken with extreme care.**

After running the commands above, you'll need to go inside the  "./build/html" folder and copy all the files.

```bash
git branch -a
# Should be something like origin/doc/ where origin is the remote to the theochem/grid Github
git checkout gh-pages
cd ..  # Get out of doc and go back to grid folder
```
Now, paste and overwrite all of the files into the root. Then commit and push to the `gh-pages` branch of the theochem/grid
```bash
git add *
git commit -m "Update website"
git push origin gh-pages
```
