# yaml

可参考：
* https://realpython.com/python-yaml/
* https://pyyaml.org/wiki/PyYAMLDocumentation

## 构建构造及表示

```python
import yaml


class B:
    def __init__(self, b1) -> None:
        self.b1 = b1

    def __repr__(self) -> str:
        return f"{__class__}(b1={self.b1})"


class A:
    def __init__(self, p1, p2, b) -> None:
        self.p1 = p1
        self.p2 = p2
        self.p3 = self.p1 + self.p2
        self.b = b


def A_representer(dumper, a):
    return dumper.represent_mapping("!A", {"p1": a.p1, "p2": a.p2, "b": a.b})


def B_representer(dumper, b):
    return dumper.represent_mapping("!B", {"b1": b.b1})


def A_constructor(loader, node):
    v = loader.construct_mapping(node, deep=True)
    return A(v["p1"], v["p2"], v["b"])


def B_constructor(loader, node):
    v = loader.construct_mapping(node, deep=True)
    return B(v["b1"])


yaml.add_representer(A, A_representer)
yaml.add_representer(B, B_representer)
yaml.add_constructor("!A", A_constructor)
yaml.add_constructor("!B", B_constructor)

a = A(3, 4, B(88))
o = {"o1": a, "o2": {"o3": a, "o4": 5}}
# o = {"o1": B(88)}

s = yaml.dump(o)
print(s)
v = yaml.load(s, yaml.UnsafeLoader)
print(v)
print(id(v["o1"]), v["o1"].p3)
print(id(v["o2"]["o3"]))
print(v["o2"]["o3"].p3)

```

另一种方式

```python
from pydoc import locate

import yaml


class Importer(yaml.YAMLObject):
    yaml_tag = "!I"

    def __init__(self, path: str):
        assert isinstance(path, str), f"导入路径必须是字符串: {path} type({type(path)})"
        self.path = path

    def __repr__(self):
        return f"<Import {self.path}>"

    @classmethod
    def from_yaml(cls, loader, node):
        v = loader.construct_scalar(node)
        return locate(v)

    @classmethod
    def to_yaml(cls, dumper, data):
        v = dumper.represent_scalar(cls.yaml_tag, data.path)
        return v


if __name__ == "__main__":

    class B:
        def __init__(self, b1) -> None:
            self.b1 = b1

        def __repr__(self) -> str:
            return f"{__class__}(b1={self.b1})"

    o = {"A": Importer("__main__.B"), "B": Importer("torch.nn.Tanh")}
    s = yaml.dump(o)
    print(s)
    o = yaml.load(s, yaml.UnsafeLoader)
    assert o["A"] == B
```
