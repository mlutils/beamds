

# import clang.cindex

# class CppModifier:
#     def __init__(self, variable_name_map, function_name_map, new_variable_types, new_function_types):
#         self.variable_name_map = variable_name_map
#         self.function_name_map = function_name_map
#         self.new_variable_types = new_variable_types
#         self.new_function_types = new_function_types
#         self.modifications = {}
#
#     def visit(self, node):
#         if node.kind == clang.cindex.CursorKind.FUNCTION_DECL:
#             self.modify_function(node)
#         elif node.kind == clang.cindex.CursorKind.VAR_DECL:
#             self.modify_variable(node)
#
#         # Recursively visit children
#         for child in node.get_children():
#             self.visit(child)
#
#     def modify_function(self, node):
#         if node.spelling in self.function_name_map:
#             # Using a tuple (start, end) as the key
#             self.modifications[(node.extent.start.offset, node.extent.end.offset)] = \
#                 self.new_function_representation(node)
#
#     def modify_variable(self, node):
#         if node.spelling in self.variable_name_map:
#             # Using a tuple (start, end) as the key
#             self.modifications[(node.extent.start.offset, node.extent.end.offset)] = \
#                 self.new_variable_representation(node)
#
#     def new_function_representation(self, node):
#         new_name = self.function_name_map.get(node.spelling, node.spelling)
#         new_return_type = self.new_function_types.get(node.spelling, None)
#
#         # Fetching the return type from the AST node
#         original_return_type = node.result_type.spelling
#
#         # Use new return type if it's provided, else use the original
#         return_type = new_return_type if new_return_type else original_return_type
#
#         # Reconstructing the function signature
#         param_list = ', '.join([param.spelling for param in node.get_arguments()])
#         return f"{return_type} {new_name}({param_list})"
#
#     def new_variable_representation(self, node):
#         new_name = self.variable_name_map.get(node.spelling, node.spelling)
#         new_type = self.new_variable_types.get(node.spelling, None)
#
#         # Fetching the type from the AST node
#         original_type = node.type.spelling
#
#         # Use new type if it's provided, else use the original
#         var_type = new_type if new_type else original_type
#
#         return f"{var_type} {new_name}"
#
#     def transform(self, code):
#         index = clang.cindex.Index.create()
#         translation_unit = index.parse("tmp.cpp", unsaved_files=[('tmp.cpp', code)], args=['-std=c++11'])
#
#         self.modifications = {}  # Resetting the modifications dictionary
#         self.visit(translation_unit.cursor)
#
#         # Sorting the modifications based on the start offset in reverse order
#         for (start, end) in sorted(self.modifications.keys(), key=lambda x: x[0], reverse=True):
#             code = code[:start] + self.modifications[(start, end)] + code[end:]
#
#         return code


from pycparserext.ext_c_parser import GnuCParser
from pycparser import c_generator, c_ast, c_parser


class CppModifier:
    def __init__(self, variable_name_map, function_name_map, new_variable_types, new_function_types):
        self.variable_name_map = variable_name_map
        self.function_name_map = function_name_map
        self.new_variable_types = new_variable_types
        self.new_function_types = new_function_types

    def visit(self, node):
        if isinstance(node, c_ast.FuncDef):
            self.modify_function(node)
        elif isinstance(node, c_ast.Decl):
            self.modify_variable(node)

        # Recursively visit children
        for child in node:
            self.visit(child)

    def modify_function(self, node):
        # Modify function name
        if node.decl.name in self.function_name_map:
            # Update the function name
            node.decl.name = self.function_name_map[node.decl.name]

        # Modify return type
        if node.decl.name in self.new_function_types:
            # Update the return type
            new_return_type = self.new_function_types[node.decl.name]
            node.decl.type.type = self.convert_type(new_return_type)

        # Modify parameter types, if parameters exist
        if node.decl.type.args:
            for param in node.decl.type.args.params:
                if param.name in self.new_variable_types:
                    # Update the parameter type
                    new_param_type = self.new_variable_types[param.name]
                    param.type = self.convert_type(new_param_type, param.name)

    def modify_variable(self, node):
        if isinstance(node.type, c_ast.TypeDecl):
            # Modify variable name
            if node.name in self.variable_name_map:
                node.name = self.variable_name_map[node.name]

            # Modify variable type
            if node.name in self.new_variable_types:
                new_var_type = self.new_variable_types[node.name]
                node.type.type = self.convert_type(new_var_type)

    def convert_type(self, new_type_str, declname=''):
        # Create a new TypeDecl node with the updated type
        return c_ast.TypeDecl(
            declname=declname,
            quals=[],
            type=c_ast.IdentifierType(names=[new_type_str])
        )

    def transform(self, code):
        parser = c_parser.CParser()
        ast = parser.parse(code)

        self.visit(ast)

        generator = c_generator.CGenerator()
        return generator.visit(ast)


if __name__ == '__main__':

    # Modifications maps
    variable_name_map = {'num1': 'firstNumber', 'num2': 'secondNumber'}
    function_name_map = {'sum': 'addNumbers', 'swap': 'exchange'}
    new_variable_types = {'firstNumber': 'long', 'secondNumber': 'long'}
    new_function_types = {'addNumbers': 'long', 'exchange': 'void'}

    modifier = CppModifier(variable_name_map, function_name_map, new_variable_types, new_function_types)

    # Your C code as a string
    code = """

    int sum(int a, int b) {
        return a + b;
    }

    void swap(double *x, double *y) {
        double temp = *x;
        *x = *y;
        *y = temp;
    }

    int main() {
        int num1 = 5, num2 = 10;
        double val1 = 3.5, val2 = 4.5;

        printf("Sum: %d\\n", sum(num1, num2));

        swap(&val1, &val2);
        printf("After swap: val1 = %lf, val2 = %lf\\n", val1, val2);

        return 0;
    }
    """

    # Apply transformations
    modified_code = modifier.transform(code)
    print(modified_code)



    # def transform(self, code):
    #     parser = GnuCParser()
    #     ast = parser.parse(code)
    #
    #     # self.visit(ast)
    #
    #     generator = c_generator.CGenerator()
    #     return generator.visit(ast)


    # code = """
    #
    # #include <iostream>
    # #include <cmath>
    # using namespace std;
    #
    # int main() {
    #
    #     int a;
    #     float b, c, x1, x2, discriminant, realPart, imaginaryPart;
    #     cout << "Enter coefficients a, b and c: ";
    #     cin >> a >> b >> c;
    #     discriminant = b*b - 4*a*c;
    #
    #     if (discriminant > 0) {
    #         x1 = (-b + sqrt(discriminant)) / (2*a);
    #         x2 = (-b - sqrt(discriminant)) / (2*a);
    #         cout << "Roots are real and different." << endl;
    #         cout << "x1 = " << x1 << endl;
    #         cout << "x2 = " << x2 << endl;
    #     }
    #
    #     else if (discriminant == 0) {
    #         cout << "Roots are real and same." << endl;
    #         x1 = -b/(2*a);
    #         cout << "x1 = x2 =" << x1 << endl;
    #     }
    #
    #     else {
    #         realPart = -b/(2*a);
    #         imaginaryPart =sqrt(-discriminant)/(2*a);
    #         cout << "Roots are complex and different."  << endl;
    #         cout << "x1 = " << realPart << "+" << imaginaryPart << "i" << endl;
    #         cout << "x2 = " << realPart << "-" << imaginaryPart << "i" << endl;
    #     }
    #
    #     return 0;
    # }
    #
    # """

    # code = """
    #
    # int main() {
    #   double first, second, temp;
    #   printf("Enter first number: ");
    #   scanf("%lf", &first);
    #   printf("Enter second number: ");
    #   scanf("%lf", &second);
    #
    #   temp = first;
    #
    #   first = second;
    #
    #   second = temp;
    #
    #   printf("\nAfter swapping, first number = %.2lf\n", first);
    #   printf("After swapping, second number = %.2lf", second);
    #   return 0;
    # }
    #
    # """

    # function_name_map = {'main': 'roots'}
    # variable_name_map = {'discriminant': 'd', 'realPart': 'rp', 'imaginaryPart': 'ip'}
    # new_variable_types = {'a': 'float'}
    # new_function_types = {'main': 'void'}
    #
    # # Example usage
    #
    # modifier = CppModifier(variable_name_map, function_name_map, new_variable_types, new_function_types)
    # modified_code = modifier.transform(code)

    # variable_name_map = {'oldVarName': 'newVarName'}
    # function_name_map = {'oldFuncName': 'newFuncName'}
    # new_variable_types = {'oldVarName': 'int'}
    # new_function_types = {'oldFuncName': 'void'}
    #
    # modifier = CppModifier(variable_name_map, function_name_map, new_variable_types, new_function_types)
    # modified_code = modifier.transform("void oldFuncName() { float oldVarName; }")
    #
    # print(modified_code)