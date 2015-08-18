import codecs
import os
import shutil
import imp
import platform

import pygeoprocessing.testing
import pygeoprocessing.geoprocessing


def file_has_class(test_file_uri, test_class_name):
    """
    Check that a python test file contains a class.

    Parameters:
        test_file_uri (string): a URI to a python file containing test classes.
        test_class_name (string): a string, the class name we're looking for.

    Returns:
        True if the class is found, False otherwise."""

    try:
        module = imp.load_source('model', test_file_uri)
    except ImportError:
        # We couldn't import everything necessary (such as with
        # invest_test_core), so we need to loop line by line to check and see
        # if the class has the test required.
        with codecs.open(test_file_uri, mode='r', encoding='utf-8') as testfile:
            for line in testfile:
                if line.startswith('class %s(' % test_class_name):
                    return True
        return False
    return hasattr(module, test_class_name)


def class_has_ftest(test_file_uri, test_class_name, test_func_name):
    """
    Check that a python test file contains the given class and function.
    This assumes that the test file is implemented using unittest.TestCase.

    Parameters:
        test_file_uri (string): a URI to a python file containing test classes.
        test_class_name (string): a string, the class name we're looking for.
        test_func_name (string): a string, the test function name we're looking
            for. This function should be located within the target test class.

    Returns:
        True if the function is found within the class, False
        otherwise."""

    try:
        module = imp.load_source('model', test_file_uri)
        print 'imported'
        cls_attr = getattr(module, test_class_name)
        print 'cls'
        return (hasattr(module, test_class_name) and
                hasattr(cls_attr, test_func_name))
    except (ImportError, AttributeError):
        # AttributeError when everythig imported, but we could not find the
        # test class and test function.  ALso happens with nested classes.
        # ImportError when we couldn't import everything necessary (such as with
        # invest_test_core), so we need to loop line by line to check and see
        # if the class has the test required.
        in_class = False
        with codecs.open(test_file_uri, mode='r', encoding='utf-8') as testfile:
            for line in testfile:
                if line.strip().startswith('class %s(' % test_class_name):
                    in_class = True
                elif in_class:
                    if line.startswith('class '):
                        # We went through the whole class and didn't find the
                        # function.
                        return False
                    elif line.lstrip().startswith('def %s(self):' %
                                                test_func_name):
                        # We found the function within this class!
                        return True
        return False


def add_ftest_to_class(file_uri, test_class_name, test_func_name,
                       in_archive_uri, out_archive_uri, module):
    """
    Add a test function to an existing test file.  The test added is a
    regression test using the pygeoprocessing.testing.regression archive
    decorator.

    Parameters:
        file_uri (string): URI to the test file to modify.
        test_class_name (string): The test class name to modify.  If the test
            class already exists, the test function will be added to the test
            class. If not, the new class will be created.
        test_func_name (string):  The name of the test function to write.  If a
            test function by this name already exists in the target class, the
            function will not be written.
        in_archive_uri (string): URI to the input archive.
        out_archive_uri (string): URI to the output archive.
        module (string): string module, whose execute function will be run in the test
            (e.g. 'natcap.invest.pollination.pollination')

    WARNING: The input test file is overwritten with the new test file.

    Returns:
        None
    """

    cls_exists = file_has_class(file_uri, test_class_name)
    test_exists = class_has_ftest(file_uri, test_class_name, test_func_name)

    if test_exists:
        print ('WARNING: %s.%s exists.  Not writing a new test.' %
               (test_class_name, test_func_name))
        return

    import_ = 'import pygeoprocessing.testing\n'
    class_ = 'class {classname}(unittest.TestCase):\n'.format(
        classname=test_class_name)

    def _archive_reg_test(test_name, module, in_archive, out_archive, cur_dir):
        """
        Format a regression archive string.

        Parameters:
            test_name (string): the name of the new test.
            module (string): the name of the module that contains the execute
                function we are calling in the test.
            in_archive (string): the path to the archive of inputs.
            out_archive (string): the path to the archive of outputs.
            cur_dir (string): the CWD.  Paths to the archives are made relative
                to this folder.

        Returns:
            A formatted string that contains the decorated test case to be
            written.
        """
        in_archive = os.path.relpath(in_archive, cur_dir)
        out_archive = os.path.relpath(out_archive, cur_dir)
        if platform.system() == 'Windows':
            in_archive = in_archive.replace(os.sep, '/')
            out_archive = out_archive.replace(os.sep, '/')

        return('    @pygeoprocessing.testing.regression(\n'
               '        input_archive="{in_arch}",\n'
               '        workspace_archive="{out_arch}")\n'
               '    def {test_name}(self):\n'
               '        {module}.execute(self.args)\n'
               '\n').format(in_arch=in_archive, out_arch=out_archive,
                            test_name=test_name, module=module)

    test_file = codecs.open(file_uri, 'r', encoding='utf-8')
    temp_file_uri = pygeoprocessing.geoprocessing.temporary_filename()
    new_file = codecs.open(temp_file_uri, 'w+', encoding='utf-8')
    if cls_exists is False:
        for line in test_file:
            new_file.write(line.rstrip() + '\n')

        new_file.write('\n')
        new_file.write(import_)
        new_file.write(class_)
        new_file.write(_archive_reg_test(test_func_name, module,
                       in_archive_uri, out_archive_uri,
                       os.path.dirname(file_uri)))
    else:
        import_written = False
        for line in test_file:
            if ((line.startswith('import') is False or line.startswith('from'))
                    and not import_written):
                new_file.write(import_)
                import_written = True

            new_file.write(line.rstrip() + '\n')
            if 'class %s(' % test_class_name in line:
                new_file.write(_archive_reg_test(test_func_name, module,
                               in_archive_uri, out_archive_uri,
                               os.path.dirname(file_uri)))

    test_file.close()
    new_file.close()

    # delete the old file
    os.remove(file_uri)
    print 'removed %s' % file_uri

    # copy the new file over the old one.
    shutil.copyfile(temp_file_uri, file_uri)
    print 'copying %s to %s' % (temp_file_uri, file_uri)
