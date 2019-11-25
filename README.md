# Objectives

The learning objectives of this assignment are to:
1. practice working with sequential part-of-speech tagging data 
2. become familiar with training maximum entropy Markov models

# Setup your environment

You will need to set up an appropriate coding environment on whatever computer
you expect to use for this assignment.
Minimally, you should install:

* [git](https://git-scm.com/downloads)
* [Python (version 3.6 or higher)](https://www.python.org/downloads/)
* [numpy (version 1.11 or higher)](http://www.numpy.org/)
* [scipy (version 1.1 or higher)](https://www.scipy.org/)
* [scikit-learn (version 0.20 or higher)](http://scikit-learn.org/)
* [pytest](https://docs.pytest.org/)

# Check out a new branch

Before you start editing any code, you will need to create a new branch in your
GitHub repository to hold your work.

1. Go to the repository that GitHub Classroom created for you,
`https://github.com/UA-LING-439-SP19/memm-classifier-<your-username>`, where
`<your-username>` is your GitHub username, and
[create a branch through the GitHub interface](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/).
You may name the branch anything you like.
2. Clone the repository to your local machine and checkout the branch you
just created:
   ```
   git clone -b <branch> https://github.com/UA-LING-439-SP19/memm-classifier-<your-username>.git
   ```
   where `<branch>` is whatever you named your branch.


# Write your code

You will implement one function, `read_ptbtagged`, and six methods of a class,
`Classifier`.
A template for each of these has been provided in the `memm.py` file.
You should read the documentation strings (docstrings) in each of methods in
that file, and implement the methods as described.
Write your code below the docstring of each method; **do not delete the
docstrings**.

The following objects and functions may come in handy:
* [sklearn.feature_extraction.DictVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html)
* [sklearn.preprocessing.LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
* [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) 
* [scipy.sparse.vstack](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.vstack.html)
* [numpy.argmax](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html)


# Test your code

Tests have been provided for you in the `test_memm.py` file.
To run all the provided tests, run the ``pytest`` script from the directory
containing ``test_memm.py``.
Initially, you will see output like:
```
============================= test session starts ==============================
...
collected 5 items                                                              

test_memm.py FFFFx                                                       [100%]

=================================== FAILURES ===================================
_____________________________ test_read_ptbtagged ______________________________

    def test_read_ptbtagged():
        # keep a counter here (instead of enumerate) in case the iterator is empty
        token_count = 0
        sentence_count = 0
>       for sentence in memm.read_ptbtagged("PTBSmall/train.tagged"):
E       TypeError: 'NoneType' object is not iterable

test_memm.py:20: TypeError
...
===================== 4 failed, 1 xfailed in 1.22 seconds ======================
```
This indicates that all tests are failing, which is expected since you have not
yet written the code for any of the methods.
Once you have written the code for all methods, you should instead see
something like:
```
============================= test session starts ==============================
...
collected 5 items                                                              

test_memm.py ...
96.5% accuracy on first 100 sentences of PTB dev
.X                                                       [100%]

===================== 4 passed, 1 xpassed in 78.59 seconds =====================
```

# Submit your code

As you are working on the code, you should regularly `git commit` to save your
current changes locally and `git push` to push all saved changes to the remote
repository on GitHub.

To submit your assignment,
[create a pull request on GitHub](https://help.github.com/articles/creating-a-pull-request/#creating-the-pull-request).
where the "base" branch is "master", and the "compare" branch is the branch you
created at the beginning of this assignment.
Then go to the "Files changed" tab, and make sure that you have only changed
the `classify.py` file and that all your changes look as you would expect them
to.
**Do not merge the pull request.**

Your instructor will grade the code of this pull request, and provide you
feedback in the form of comments on the pull request.

# Grading

Assignments will be graded primarily on their ability to pass the tests that
have been provided to you.
Assignments that pass all but the last test will receive at least 80% of the
possible points.

The final test (marked with `@pytest.mark.xfail`) is optional.
If you succeed in making all tests (including this one) pass, you will receive
at least 90% of the possible points.

To get the remaining 10-20% of the points, make sure that your code is using
appropriate data structures, existing library functions are used whenever
appropriate, code duplication is minimized, variables have meaningful names,
complex pieces of code are well documented, etc.
