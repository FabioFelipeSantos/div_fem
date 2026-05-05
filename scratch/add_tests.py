from unittest.mock import MagicMock

def test_element_2d_properties():
    p1 = Point([0.0, 0.0])
    p2 = Point([10.0, 0.0])
    pts = Points(2)
    pts.add([p1, p2])
    mat = {"E": 200e9, "A": 0.01, "I": 0.0001}
    elem = Element2D((p1, p2), material_and_section_properties=mat, type="beam")
    
    assert elem.points == [p1, p2]
    assert elem.degree_of_freedom == [1, 2, 3, 4]

def test_element_2d_integration_errors_bar():
    p1 = Point([0.0, 0.0])
    p2 = Point([10.0, 0.0])
    pts = Points(1)
    pts.add([p1, p2])
    mat = {"E": 1.0, "A": 1.0, "I": 1.0}
    elem = Element2D((p1, p2), material_and_section_properties=mat, type="bar")
    
    mock_load = MagicMock(spec=Element2DLoads)
    mock_load.force_point = None
    mock_load.force_init_point = None
    mock_load.force_final_point = None
    
    # Missing force_x
    mock_load.force_type = "constant"
    mock_load.force_value.return_value = None
    elem._loads = [mock_load]
    with pytest.raises(ValueError, match="To apply loads on an bar element, provide a float or Callable"):
        _ = elem.local_forces_vector
        
    # Constant not float
    mock_load.force_value.return_value = "not float"
    with pytest.raises(ValueError, match="Provide a valid float for constant distributed load value for the x axis for elements of type bar."):
        _ = elem.local_forces_vector
        
    # Function not callable
    mock_load.force_type = "function"
    with pytest.raises(ValueError, match="For a function force in a bar, provide a callable"):
        _ = elem.local_forces_vector

def test_element_2d_integration_errors_beam():
    p1 = Point([0.0, 0.0])
    p2 = Point([10.0, 0.0])
    pts = Points(2)
    pts.add([p1, p2])
    mat = {"E": 1.0, "A": 1.0, "I": 1.0}
    elem = Element2D((p1, p2), material_and_section_properties=mat, type="beam")
    
    mock_load = MagicMock(spec=Element2DLoads)
    mock_load.force_point = None
    mock_load.force_init_point = None
    mock_load.force_final_point = None
    
    # Missing force_y and moment
    mock_load.force_type = "constant"
    mock_load.force_value.return_value = (None, None)
    elem._loads = [mock_load]
    with pytest.raises(ValueError, match="To apply loads on an bar element, provide a float or Callable"):
        _ = elem.local_forces_vector
        
    # Constant force_y not float
    mock_load.force_value.return_value = ("not float", None)
    with pytest.raises(ValueError, match="Provide a valid float for constant load values for the y axis for elements of type beam."):
        _ = elem.local_forces_vector
        
    # Constant moment not float
    mock_load.force_value.return_value = (None, "not float")
    with pytest.raises(ValueError, match="Provide a valid float for constant load values for moment for elements of type beam."):
        _ = elem.local_forces_vector
        
    # Function force_y not callable
    mock_load.force_type = "function"
    mock_load.force_value.return_value = ("not callable", None)
    with pytest.raises(ValueError, match="For a function force in y axis, provide a callable"):
        _ = elem.local_forces_vector
        
    # Function moment not callable
    mock_load.force_value.return_value = (None, "not callable")
    with pytest.raises(ValueError, match="For a function force in moment, provide a callable"):
        _ = elem.local_forces_vector

def test_element_2d_integration_errors_frame():
    p1 = Point([0.0, 0.0])
    p2 = Point([10.0, 0.0])
    pts = Points(3)
    pts.add([p1, p2])
    mat = {"E": 1.0, "A": 1.0, "I": 1.0}
    elem = Element2D((p1, p2), material_and_section_properties=mat, type="frame")
    
    mock_load = MagicMock(spec=Element2DLoads)
    mock_load.force_point = None
    mock_load.force_init_point = None
    mock_load.force_final_point = None
    
    # Missing all
    mock_load.force_type = "constant"
    mock_load.force_value.return_value = (None, None, None)
    elem._loads = [mock_load]
    with pytest.raises(ValueError, match="To apply loads on an bar element, provide a float or Callable"):
        _ = elem.local_forces_vector
        
    # Constant not float
    mock_load.force_value.return_value = ("not float", None, None)
    with pytest.raises(ValueError, match="Provide a valid float for constant load values for the x axis for elements of type frame."):
        _ = elem.local_forces_vector
    mock_load.force_value.return_value = (None, "not float", None)
    with pytest.raises(ValueError, match="Provide a valid float for constant load values for the y axis for elements of type frame."):
        _ = elem.local_forces_vector
    mock_load.force_value.return_value = (None, None, "not float")
    with pytest.raises(ValueError, match="Provide a valid float for constant load values for moment for elements of type frame."):
        _ = elem.local_forces_vector
        
    # Function not callable
    mock_load.force_type = "function"
    mock_load.force_value.return_value = ("not callable", None, None)
    with pytest.raises(ValueError, match="For a function force in x axis, provide a callable"):
        _ = elem.local_forces_vector
    mock_load.force_value.return_value = (None, "not callable", None)
    with pytest.raises(ValueError, match="For a function force in y axis, provide a callable"):
        _ = elem.local_forces_vector
    mock_load.force_value.return_value = (None, None, "not callable")
    with pytest.raises(ValueError, match="For a function force in moment, provide a callable"):
        _ = elem.local_forces_vector

def test_element_2d_concentrated_errors():
    p1 = Point([0.0, 0.0])
    p2 = Point([10.0, 0.0])
    pts = Points(3)
    pts.add([p1, p2])
    mat = {"E": 1.0, "A": 1.0, "I": 1.0}
    
    # Beam concentrated errors
    elem_beam = Element2D((p1, p2), material_and_section_properties=mat, type="beam")
    mock_load = MagicMock(spec=Element2DLoads)
    mock_load.force_type = "concentrated"
    mock_load.force_point = 0.5
    
    mock_load.force_value.return_value = (None, None)
    elem_beam._loads = [mock_load]
    with pytest.raises(ValueError, match="To concentrated loads in beams, provide a value for y axis or for moment."):
        _ = elem_beam.local_forces_vector
        
    mock_load.force_type = "moment"
    with pytest.raises(ValueError, match="To concentrated moment in beams, provide a float value for the moment."):
        _ = elem_beam.local_forces_vector
        
    mock_load.force_type = "concentrated"
    mock_load.force_value.return_value = ("not float", None)
    with pytest.raises(ValueError, match="To concentrated y axis load provide a float value."):
        _ = elem_beam.local_forces_vector
        
    mock_load.force_value.return_value = (None, "not float")
    with pytest.raises(ValueError, match="To concentrated y axis load provide a float value."):
        _ = elem_beam.local_forces_vector

    # Frame concentrated errors
    elem_frame = Element2D((p1, p2), material_and_section_properties=mat, type="frame")
    mock_load.force_type = "concentrated"
    mock_load.force_value.return_value = (None, None, None)
    elem_frame._loads = [mock_load]
    with pytest.raises(ValueError, match="To concentrated loads in frames, provide a value for x or y axis or for moment."):
        _ = elem_frame.local_forces_vector
        
    mock_load.force_type = "moment"
    with pytest.raises(ValueError, match="To concentrated moment in frames, provide a float value for the moment."):
        _ = elem_frame.local_forces_vector
