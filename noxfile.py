import nox

@nox.session(python="3.9")
def tests(session):
    session.install(".")
    session.install("pytest")
    
    if session.posargs:
        test_files = session.posargs
    else:
        test_files = ['speakeazy/tests/test_sampling.py']

    session.run('pytest', *test_files)