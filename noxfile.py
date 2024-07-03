"""
Creation:
    Author: Martin Grunnill
    Date: 2024-07-03
Description:
    Noxfile for automating:
        - Unit Tests
"""
import nox
import pathlib

@nox.session(reuse_venv=True)
def tests(session):
    """Run test cases."""
    session.install('.')
    test_dir = pathlib.Path('tests')
    for test_file in test_dir.glob('test_*.py'):
        session.run('python', test_file)