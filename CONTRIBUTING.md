## How to contribute
Welcome to Grid project by Ayers Group. Thank you for taking your time reading this guidlines

**`Note`**: all commands embraced by `{}` need to be placed by specific varible

#### New to GitHub

* Fork the repo by clicking the `fork` button on the top right screen. This will create a copy of this project in your own account.
* Go to your forked repo mainpage. Click the green `Clone or download` button, copy the link, and type the command in your terminal with the link you just copied:
```bash
    git clone git@github.com:{your_user_name}/grid.git
```
or
```bash
    git clone https://github.com/{your_user_name}/grid.git
```
This command will download the code to your present directory in your local machine.

#### Make changes
* You are free to make any changes in the code you just downloaded. The default branch is `master`. We highly recommand you createing a new branch for developing new functionalities. To do so,
```bash
    git checkout -b {new_branch_name}
```

<!-- * When the changes is ready, add the modified file, and make a commit
```bash
    git add {modified_filename}
    git commit -m "{commit messgae for the changes}"
``` -->
* In this new branch, you are free to explore all kinds of new ideas; make any changes you'd like.

* If you refactored or modify some code blocks, it is important that all the code still work as intented.
To do so, make sure you have `tox` installed in your environment, then run
```bash
    tox
```
This command will trigger `tox` to execute all the test environments includeing `py36`, `py37`, `black` style, `flake` style tests.
You can also invoke a specific test env by
```bash
    tox -e {env_name}
```

* If you added some new functionalities to Grid, it would be a good idea to add tests for your block of code.
To do so, you need can create a file in `grid/src/grid/tests/` folder. Simply name the filename starts with `test_`
We adopted `unittest` style of tests, but it's compatible with all popular test frame such as `pytest` and `nosetests`.

* When all changes are ready, add the modified file, and make a commit
```bash
    git add {modified_filenames}
    git commit -m "{commit messgae for the changes}"
```

#### Contribute to Grid

* If you have all the commits ready, it's time to push your latest modified code to your remote repo by
```bash
    git push origin {branch_name}
```


* In the browser, navigate to your GitHub repo, click the `New pull request` button, select the branch you just updated.
This would create a `Pull Request` to the original repo. The maintainer will review your changes very quickly. When it's all ready, the changes will be merged back to `Grid`.

#### Bug reposrts, function request, other techinical problems

* If you encounter any bug in `Grid`, please file an [issue](https://github.com/theochem/grid/issues) with the error message, working environment and error command. We deeply appreciate your help.

* If you request some functionality, please file an [issue](https://github.com/theochem/grid/issues) with detailed description, scientific reference.

* If you need any technical support for using `Grid`, please file an [issue](https://github.com/theochem/grid/issues) with your detailed question and your working environment.

Thanks! :heart: :heart: :heart:

Ayers Group
