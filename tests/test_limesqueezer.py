
from click.testing import CliRunner

from limesqueezer.cli import main


def test_main():
    runner = CliRunner()
    result = runner.invoke(main, [])

    assert result.output == '()\n'
    assert result.exit_code == 0

# import reference
# import plotters

# data = reference.generate('sine', 'LSQ10')

# plotters.plot_1_data_compressed(data)