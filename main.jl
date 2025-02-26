using SealSolver


function write_to_log(msg, ex)

    open("julia_logs.log", "a") do io
        println(io, msg)
        println(io, "Error: ", string(ex))
        bt = catch_backtrace()
        Base.show_backtrace(io, bt)
        println(io, "\n----------------------------------------")
    end

end

function prime_solver()
    # compute and discard the results
    duration = @elapsed begin
        res = SealSolver.prime("quick")
    end

    println("Primed in <$duration> seconds")
end

function save_results(data::Array{Tuple{String,Array}}, folder::String)
    for pair in data
        k, v = pair

        file_name = joinpath(folder, "$(k).h5")
        h5open(file_name, "w") do fid
            write(fid, k, v)
        end

    end
end


function run_solver(params_path::String, results_path::String, csv_file::String)
    try
        @info "Solving on path <$params_path>"
        res = []
        duration = @elapsed begin
            res = SealSolver.solve(params_path)
        end
        if length(res) == 0
            error("Computation Failed. <res> cannot be nothing. See logs for failure reasons")
        end

        save_duration = @elapsed begin
            save_results(res, results_path)
        end

        total_duration = duration + save_duration

        # append_to_csv((duration, save_duration, total_duration), csv_file)

        println("------------------------------------------")
        println("Complete")
        println("------------------------------------------")
        println("\tCompute time => <$duration secs>")
        println("\tSave time => <$save_duration secs>")
        println("\t total time => <$(total_duration) secs>")
        println("------------------------------------------\n")
    catch ex
        @warn "Run failed. See log.log file"
        msg = "Run Failed:\n\t[PARAMS_FOLDER] => $params_path"
        write_to_log(msg, ex)
        # Base.rethrow(ex)
    end
end

function append_to_csv(values::Tuple{Float64,Float64,Float64}, file::String)

    computeTime, saveTime, totalTime = values

    # Open the file in append mode and write the values as a new row
    open(file, "a") do io
        println(io, "$computeTime,$saveTime,$totalTime")
    end
end

function run_multiple_solver(folders::Vector{Tuple{String,String}}, csv_file::String)
    i = 0
    for pair in folders
        params_folder, results_folders = pair
        run_solver(params_folder, results_folders, csv_file)

        i = i + 1
        percentage = round((i / length(folders)) * 100, digits=2)
        println("\n Progression: <$percentage%>")
        println("=========================================")
    end
end

# create the base simulation case folders

case_keys = [
    # "case_1",
    # "case_2",
    "case_3",
    "case_4",
]

grid_keys = [
    # "grid_1",
    # "grid_2",
    # "grid_3",
    "grid_4",
]


cases = []

for grid_key in grid_keys
    for case_key in case_keys
        push!(cases, "$(grid_key)-$(case_key)")
    end
end


FILE_FOLDERS::Vector{Tuple{String,String}} = []

TEST_FOLDERS::Vector{Tuple{String,String}} = [(
    joinpath(pwd(), "test_case", "params"),
    joinpath(pwd(), "test_case", "results"),
)]



CSV_FILE = joinpath(pwd(), "durations.csv")

for case in cases
    pair = (
        joinpath(pwd(), case, "params"),
        joinpath(pwd(), case, "results"),
    )
    push!(FILE_FOLDERS, pair)
end


function run_complete_set()
    @info "Starting set"

    run_multiple_solver(FILE_FOLDERS, CSV_FILE)

    @info "Set Complete"
end

function run_test_set() 
    @info "Starting Test"

    run_multiple_solver(TEST_FOLDERS, CSV_FILE)
    
    @info "Test Complete"
end

println("Ready for usage")
println("call prime_solve() to prepare system")
println("call <run_test_set()> to run a test")
println("call <run_complete_set()>")
