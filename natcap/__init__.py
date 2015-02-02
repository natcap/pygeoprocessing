import pkg_resources
pkg_resources.declare_namespace(__name__)

from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)
