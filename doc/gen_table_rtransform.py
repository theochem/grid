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


def generate_rtransform_grid_table_csv():
    r"""
    Parse the rtransform.py file for all subclasses of BaseTransform,
    then the next line should have a from and to that tells you the domain and codomain of
    transformation.
    This creates a .csv file which is then added to index.rst.

    This parses "Grid Name" from the documentation line.
    class Transformation(BaseTransform):
        "Some Transformation from :math:`[-1, 1]` to :math:`[a, b]`. "
    The words "from" and "to" must seperate the name or else this will fail.

    In order to parse the transformation it assumes the following format:
    def transform(self, npoints):
       r\"\"\"
       Blah blah

       .. math::
            r_i = trnasfomration here.
        \"\"\"

    """
    df = pd.DataFrame(columns=["Transform", "Domain", "Co-Domain", ":math:`r(x)`"])
    with open(r"../src/grid/rtransform.py", "r") as f:
        line = f.readline()
        found_class = False
        while line:
            line = line.rstrip()  # strip trailing spaces and newline
            print("line: ", line)
            # If you found a class that is a child of BaseTransform, then explore further.
            if "InverseRTransform" in line and "BaseTransform" in line:
                row_result = []
                row_result += [
                    ":func:`InverseRTransform<grid.rtransform.InverseRTransform>`",
                    "Co-Domain",
                    "Domain",
                    ":math:`x_i(r_i)",
                ]
                print(row_result)
                df.loc[len(df.index)] = row_result
            elif "IdentityRTransform" in line and "BaseTransform" in line:
                row_result = []
                row_result += [
                    ":func:`IdentityRTransform<grid.rtransform.IdentityRTransform>`",
                    "Domain",
                    "Domain",
                    ":math:`x_i`",
                ]
                df.loc[len(df.index)] = row_result
            elif (
                "class" in line
                and "BaseTransform" in line
                and "issubclass" not in line
                and "ABC" not in line
            ):
                row_result = []
                # Get the class name, by looking at what's after class
                class_name = line.split("class")[1]
                # Remove the OneDGrid
                class_name = class_name.split("(BaseTransform)")[0].strip()

                # Get the name of the RTransform
                line_document = f.readline()
                print("linedoc", line_document.strip())
                if line_document.strip() == 'r"""':
                    line_document = f.readline()
                    print("linedoc rev", line_document)
                print(class_name, line_document)

                # Add the correct link to the class name
                row_result.append(":func:`" + class_name + "<grid.rtransform." + class_name + ">`")

                if found_class:
                    raise RuntimeError(
                        f"Found a class that it could not parse, it is the class before {line}."
                    )

                found_class = True

                # Get the domain and codomain information.
                # The next line should contain the domain information
                line_start = line_document  # f.readline()
                print(line_start)
                if line_start.strip() == 'r"""':
                    line_start = f.readline()
                    print(line_start)
                if "from" not in line_start or "to" not in line_start:
                    raise RuntimeError(
                        f"The words 'from' and 'to' needs to be in {line_start} for this to parse correctly."
                    )

                # Line should be from domain to
                # Seperat ebased on "to"
                print(line_start.split("to"))
                line_domain = line_start.split("to")[0].split("from")[1].strip()
                line_codomain = line_start.split("to")[1].strip()
                print("Line domain and codomain", line_domain, line_codomain)

                # Remove white-space
                line_domain = line_domain.replace(" ", "")
                line_codomain = line_codomain.replace(" ", "")

                # Remove period
                line_codomain = line_codomain.replace(".", "")
                # Store the result into pandas
                row_result.append(str(line_domain))
                row_result.append(str(line_codomain))

            # If found a class then search for domain key.
            if found_class:
                # Get the transformation.
                if "transform(self," in line:
                    # Get the transformation from latex
                    line_transf = f.readline()
                    while ".. math::" not in line_transf:  # Keep going until you see  .. math::
                        line_transf = f.readline()
                    line_transf = f.readline().strip()  # Get the first line of transformation

                    print("Transformation ", line_transf)
                    row_result.append(":math:`" + str(line_transf) + "`")

                    df.loc[len(df.index)] = row_result
                    # Move to the next class
                    found_class = False

            line = f.readline()
    print(df)
    df.to_csv("table_rtransform.csv", index=False, sep=";")


if __name__ == "__main__":
    generate_rtransform_grid_table_csv()
