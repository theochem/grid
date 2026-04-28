# Instructions On Documentation

In-order to build the documentation locally, the following commands need to
be runned. The html files are created in an folder called "_build".

```bash
# # Generates API html for grid while ignoring the test and data folders.
# Stores it in pyapi/
sphinx-apidoc --separate -o website/_autosummary grid -M -f
# Build the html files
jupyter-book build ./website/
```
Here inside the "./website/_build/html/ folder, there are the html files to run the website locally.


## Pushing To Website

After running the commands above, you'll need to go inside the  "./website/_build/html" folder and copy all the files.

```bash
git branch -a
# Should be something like origin/gh-pages/ where origin is the remote to the theochem/gbasis Github
git checkout gh-pages
cd ..  # Get out of gh-pages and go back to gbasis folder
```
Now, paste and overwrite all of the files in gh-pages with the html. Then commit and push to the `gh-pages` branch of the theochem/gbasis
```bash
git add ./*
git commit -m "Update website"
git push origin gh-pages
```
