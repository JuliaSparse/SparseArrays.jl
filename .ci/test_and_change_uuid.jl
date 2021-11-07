@static if Base.VERSION >= v"1.6"
    using TOML
    using Test
else
    using Pkg: TOML
    using Test
end

# To generate the new UUID, we simply modify the first character of the original UUID
const original_uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
const new_uuid      = "1f01184e-e22b-5df5-ae63-d93ebab69eaf"

# `@__DIR__` is the `.ci/` folder.
# Therefore, `dirname(@__DIR__)` is the repository root.
const project_filename = joinpath(dirname(@__DIR__), "Project.toml")

@testset "Test that the UUID is unchanged" begin
    project_dict = TOML.parsefile(project_filename)
    @test project_dict["uuid"] == original_uuid
end

write(
    project_filename,
    replace(
        read(project_filename, String),
        r"uuid = .*?\n" => "uuid = \"$(new_uuid)\"\n",
    ),
)
