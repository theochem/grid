# GRID is a numerical integration module for quantum chemistry.
#
# Copyright (C) 2011-2019 The GRID Development Team
#
# This file is part of GRID.
#
# GRID is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# GRID is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
# --
r"""
Create a table of one-dimensional grids by parsing the grid data folder.
This creates a "table_onedgrids.csv" file which is then added to index.rst file.

"""
import pandas as pd


def generate_one_dimensional_grid_table_csv():
    r"""
    Parse the onedgrid.py file for all subclasses of OneDGrid, then find the __init__ line that contains domain info.
    This creates a .csv file which is then added to index.rst.

    This parses "Grid Name" from the documentation line.
    class GridName():
        "Grid Name integral quadrature class. "
    The words "Integral" must seperate the name or else this will fail.

    In order to parse the domain it assumes the following format:
    def __init__(self, npoints):
        r/"/"/"Generate 1-D grid on [l_bnd, u_bnd] interval using Interior Rectangle Rule for Sines.
    The words on and interval should seperate the domain information.

    """
    df = pd.DataFrame(columns=["Grid", "Domain"])
    with open(r"../src/grid/onedgrid.py", "r") as f:
        line = f.readline()
        found_class = False
        while line:
            line = line.rstrip()  # strip trailing spaces and newline
            print("line: ", line)
            # If you found a class that is a child of OneDGrid, then explore further.
            if "class" in line and "OneDGrid" in line and "issubclass" not in line:
                row_result = []

                # Get the class name, by looking at what's after class
                class_name = line.split("class")[1]
                # Remove the OneDGrid
                class_name = class_name.split("(OneDGrid)")[0].strip()

                # Get the name of the Grid
                line_document = f.readline()
                print("linedoc", line_document.strip())
                if line_document.strip() == 'r"""':
                    line_document = f.readline()
                    print("linedoc rev", line_document)
                if "integral" in line_document:
                    line_document = line_document.split("integral")
                    line_name = line_document[0].strip()
                    # Remove triple quotation marks: """
                    line_name = line_name[:]
                    line_name = line_name.replace('r"""', "")
                    line_name = line_name.replace('"""', "")

                    if "HORTON" in line_name:
                        line_name = line_name.replace("HORTON", "Horton")
                    # Store it in row_result plus the link to the funciton
                    row_result.append(":func:`" + line_name + "<grid.onedgrid." + class_name + ">`")
                else:
                    raise RuntimeError(
                        f"The word integral needs to be in the document, instead it is {line_document}."
                    )

                if found_class:
                    raise RuntimeError(
                        f"Found a class that it could not parse, it is the class before {line}."
                    )

                found_class = True

            # If found a class then search for domain key.
            if found_class:
                if "def __init__" in line:
                    # The next line should contain the domain information
                    line_domain = f.readline()
                    print(line_domain)

                    if line_domain.strip() == 'r"""':
                        line_domain = f.readline()
                        print(line_domain)
                    if "on" not in line_domain:  # or "interval" not in line_domain:
                        raise RuntimeError(
                            f"The words 'on' and 'interval' needs to be in {line_domain} for this to parse correctly."
                        )

                    # Seperate based on "on"
                    line_domain = line_domain.split("on")[1]

                    # Seperat ebased on "interval"
                    line_domain = line_domain.split("interval")[0].strip()

                    if "npoints" in line_domain:
                        line_domain = line_domain.replace("npoints", "N")
                    # Typo that too lazy to do pull request
                    # if ":" in line_domain:
                    #     line_domain = line_domain.replace(":", ",")

                    line_domain = line_domain.replace(" ", "")
                    # Store the result into pandas
                    row_result.append(str(line_domain))
                    df.loc[len(df.index)] = row_result
                    # Move to the next class
                    found_class = False

            line = f.readline()
    print(df)
    df.to_csv("table_onedgrids.csv", index=False, sep=";")


if __name__ == "__main__":
    generate_one_dimensional_grid_table_csv()
