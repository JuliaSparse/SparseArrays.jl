function is_github_actions_ci()
    is_ci  = parse(Bool, get(ENV, "CI",             "false"))
    is_gha = parse(Bool, get(ENV, "GITHUB_ACTIONS", "false"))

    return is_ci && is_gha
end
