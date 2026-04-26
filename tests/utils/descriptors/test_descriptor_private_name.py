import pytest
from div_fem.utils.descriptors.descriptor_private_name import DescriptorBaseClass

def validate_positive(value):
    if value <= 0:
        raise ValueError("Value must be positive")

class DummyClass:
    # Sem validação
    attr_no_val = DescriptorBaseClass()
    
    # Com validação estrita
    attr_with_val = DescriptorBaseClass(validation=validate_positive)

def test_descriptor_get_attribute_error():
    dummy = DummyClass()
    
    # Testa o erro de tentar acessar um descritor que ainda não teve valor definido
    with pytest.raises(AttributeError, match="The attr_no_val attribute hasn't have your value set. Provide a correct place to the attribute in the DummyClass class"):
        _ = dummy.attr_no_val

def test_descriptor_set_success():
    dummy = DummyClass()
    dummy.attr_no_val = 42
    assert dummy.attr_no_val == 42
    
    # Verifica se o private name (_attr_no_val) foi gerado e setado corretamente
    assert dummy._attr_no_val == 42

def test_descriptor_set_validation_error():
    dummy = DummyClass()
    
    # Testa se a exceção da validação é re-lançada corretamente pelo descritor
    with pytest.raises(ValueError, match="Value must be positive"):
        dummy.attr_with_val = -10

def test_descriptor_set_validation_success():
    dummy = DummyClass()
    dummy.attr_with_val = 15
    assert dummy.attr_with_val == 15
