def _import_analyses():
    import importlib
    import pkgutil
    import sys

    pkg = sys.modules[__name__]
    classes = {}

    for module_finder, name, ispkg in pkgutil.iter_modules(pkg.__path__):
        module_name = ".".join([pkg.__name__, name])
        try:
            module = importlib.import_module(module_name)
            classes.update({
                cls: getattr(module, cls)
                for cls in dir(module)
                if cls.endswith("Analysis")
            })
        except Exception as e:
            print('{} analysis not available: {}'.format(name, e))
    return classes


analysis_dict = _import_analyses()
__all__ = list(analysis_dict.keys())
