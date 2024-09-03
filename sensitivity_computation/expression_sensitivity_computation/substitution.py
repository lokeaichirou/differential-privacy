import ast
import astunparse
import sys
from query_errors import *


class VariableVisitor(ast.NodeTransformer):
    def set_substitution_map(self, subst_map):
        self.substitution_map = subst_map

    def node_change(self, node):
        if float(sys.version[:3]) <= 3.7:
            return ast.Num(Constant=self.substitution_map[node.id])
        else:
            return ast.Constant(self.substitution_map[node.id])

    def visit_Name(self, node):
        """

        :param node: ast.Name
        :return: None
        """
        if isinstance(node, ast.Name):
            if node.id in self.substitution_map:
                if isinstance(self.substitution_map[node.id], float) or isinstance(self.substitution_map[node.id], int):
                    node = self.node_change(node)
                elif isinstance(self.substitution_map[node.id], str):
                    node.id = self.substitution_map[node.id]
                else:
                    raise QueryException(UNSUPPORTED_SUBSTITUTION)
        self.generic_visit(node)
        return node


def subst(expr, subst_map):
    """

    :param expr:
    :param subst_map: a list of substitutions (variable_name, value), {"x":"y", "y":"x"}
    :return:
    """

    ast_parse = ast.parse(expr)
    def_change = VariableVisitor()
    def_change.set_substitution_map(subst_map)
    def_change.visit(ast_parse)
    new_expr = astunparse.unparse(ast_parse)
    print(new_expr)

    return ast_parse, new_expr


subst(expr="x+y", subst_map={"x":"y"})
subst(expr="x+y", subst_map={"x":2})
subst(expr="x**2 + y**4", subst_map={"x":"y", "y":"x"})
