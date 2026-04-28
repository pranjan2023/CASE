def observe(state):

    print("Observe stage")

    if state.history:

        last = state.history[-1]

        print("Last experiment geometry:", last["geometry"])
        print("Material:", last["geometry"].get("material"))
        print("Last experiment mean:", last["mean"])
        print("Last experiment std:", last["std"])
        print("Last experiment n:", last["n"])

    return state