using Clapeyron

function calculate_activity_coefficients(model, comps, pressure, temperature, molefraction)
    # Create an instance of the thermodynamic model with Clapeyron.
    thermo_model = Symbol(model)
    model_constructor = eval(thermo_model)
    my_mixture = model_constructor(comps)
    return activity_coefficient(my_mixture, pressure, temperature, [molefraction, 1 - molefraction])
end
