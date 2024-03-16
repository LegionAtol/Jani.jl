export parse_neurochem_resources

using HTTP
using ZipFile
using FilePaths
using Printf

const SUPPORTED_INFO_FILES = ["ani-1ccx_8x.info", "ani-1x_8x.info", "ani-2x_8x.info"]

function parse_neurochem_resources(info_file_path::AbstractString)
    jani_dir = abspath(dirname(dirname(@__FILE__)))
    resource_path = joinpath(jani_dir, "resources")
    println(resource_path)
    local_dir = joinpath(homedir(), ".local", "src")

    resource_info = joinpath(resource_path, info_file_path)
    
    if isfile(resource_info) && stat(resource_info).size > 0
        # No action needed if the info file can be located in the default path
    #TODO: after uncompress() to be tested    
    elseif isfile(joinpath(local_dir, info_file_path))
        # if the info file is not located in the default path, ~/.local/src
        # is tried as an alternative
        resource_path = local_dir
    else
        # if all else fails files are downloaded and extracted ONLY if a
        # correct info file path is passed, otherwise an error is raised
        if info_file_path in SUPPORTED_INFO_FILES
            repo_name = "ani-model-zoo"
            tag_name = "ani-2x"
            extracted_name = "$repo_name-$tag_name"
            url = "https://github.com/aiqm/$repo_name/archive/$tag_name.zip"

            println("Downloading ANI model parameters ...")
            resource_res = HTTP.get(url)
            resource_zip = ZipFile.Reader(IOBuffer(resource_res.body))
            try
                uncompress(resource_zip, resource_path)
            catch e
                uncompress(resource_zip, local_dir)
                resource_path = local_dir
            end
            source = joinpath(resource_path, extracted_name, "resources")
            cp(source, resource_path; remove_destination = true)
            rm(joinpath(resource_path, extracted_name); force = true, recursive = true)
        else
            error("File $info_file_path could not be found either in $resource_path or $local_dir\n
                  It is also not one of the supported builtin info files: $(join(SUPPORTED_INFO_FILES, ", "))")
        end
    end
    
    return _get_resources(resource_path, info_file_path)
end

function _get_resources(resource_path::AbstractString, info_file::AbstractString)
    lines = readlines(joinpath(resource_path, info_file))
    const_file_path, sae_file_path, ensemble_prefix_path, ensemble_size = lines
    const_file = joinpath(resource_path, const_file_path)
    sae_file = joinpath(resource_path, sae_file_path)
    ensemble_prefix = joinpath(resource_path, ensemble_prefix_path)
    ensemble_size = parse(Int, ensemble_size)
    return const_file, sae_file, ensemble_prefix, ensemble_size
end
