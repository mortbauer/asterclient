from itertools import product

class Variant(dict):
    def __init__(self,variant,default):
        super(Variant,self).__init__(default)
        self.update(variant)

class Variants(object):
    def __init__(self,variants,name,default=None,selected='all',deselected=[]):
        self._variants = variants
        self._default = default
        self.name = name
        self._all = set(self._variants.keys())
        self._selected = None
        self._deselected = None
        self.select(selected)
        self.deselect(deselected)

    def copy(self):
        return Variants(
            self._variants,self.name,self._default,self._selected,
            self._deselected
        )

    def select(self,variants):
        if variants == 'all':
            variants = set(self._variants.keys())
        else:
            variants = set(variants)
        if variants != self._deselected:
            self._selected = variants
        else:
            raise Exception('can\'t select and deselect same variants')

    def deselect(self,variants):
        variants = set(variants)
        if variants == self._all:
            raise Exception('you can\'t deselect everyting')
        if variants != self._selected:
            self._deselected = variants
        else:
            raise Exception('can\'t select and deselect same variants')

    def __iter__(self):
        for key,value in self._variants.items():
            if key in self._selected and key not in self._deselected:
                yield (key,self._variant(value))

    def _variant(self,variant):
        if type(variant)==dict:
            return Variant(variant,self._default)
        else:
            return variant

    def variant(self,name):
        if name in self._variants:
            variant = self._variant(self._variants[name])
        else:
            raise Exception('there is no variant "{0}"'.format(name))

class Data(object):
    def __init__(self,data):
        self._data = data
        self._variants = {}

    def multiply(self,data=None,keys=None,select={},deselect={},name=[]):
        variants = []
        if data == None:
            data = self._data
        if keys == None:
            keys = data.keys()
        subyield = False
        for key in keys:
            val = data[key]
            if isinstance(val,Variants):
                value = val.copy()
                if value.name in select:
                    value.select(select[value.name])
                elif select == 'all':
                    value.select(select)
                if value.name in deselect:
                    value.deselect(deselect[value.name])
                variants.append(product((key,),value))

        for perm in product(*variants):
            tmp = data.copy()
            name = []
            for key,value in perm:
                tmp[key] = value[1]
                name.append(value[0])
            yield name,tmp
