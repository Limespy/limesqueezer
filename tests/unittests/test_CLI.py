import pytest
from limesqueezer import CLI

def test_import__main__():
    from limesqueezer import __main__

@pytest.mark.parametrize('args', ([],
                                  ['block'],
                                  ['stream'],
                                  ['both']))
def test_main(args):
    CLI.main(args)
