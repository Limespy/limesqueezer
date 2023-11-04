import limedev
from limedev import readme
from limedev.readme import md
#=======================================================================
NAME = 'LimeDev'
#=======================================================================
def main(project_info):
    """This gets called by the limedev."""

    semi_description = md.Document([
        f'{NAME} is collection tools for Python development.\n'
        'These tools are more or less thin wrappers around other packages.'
    ])
    return readme.make(limedev, semi_description,
                       name = NAME)
#=======================================================================
