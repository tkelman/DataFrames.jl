# Formulas for representing and working with linear-model-type expressions
# Harlan D. Harris

# we can use Julia's parser and just write them as expressions

## Allow for one-sided or two-sided formulas

type Formula
    lhs::Union(Symbol,Expr,Nothing)
    rhs::Union(Symbol,Expr)
end

function Formula(ex::Expr) 
    aa = ex.args
    if ex.args[1] != :(~) error("Invalid formula, lacks '~'") end
    if length(aa) == 2 return Formula(nothing, ex.args[2]) end
    Formula(ex.args[2], ex.args[3])
end

function show(io::IO, f::Formula)
    print(io, string("Formula: ", f.lhs, " ~ ", f.rhs))
end

function all_vars(ex::Expr)             # patterned after all.vars in R
    [[all_vars(a) for a in ex.args[2:end]]...]
end

all_vars(sym::Symbol) = [sym]
all_vars(symv::Vector{Symbol}) = symv
all_vars(v) = Array(Symbol,0)

type ModelFrame
    df::AbstractDataFrame
    intercept::Bool
    order::Vector{Int}
    variables::Vector{Symbol}
    response::Int
    formula::Formula
    factors::Vector{Bool}
end

## Make the na handler an argument when named arguments are available
## Need to add a logical index extractor for DataFrame

function na_omit(df::DataFrame)
    msng = vec(reducedim(|, isna(df), [2], false))
    df[!msng,:], msng
end

## Extract the terms and orders from the rhs.
function terms(ex::Expr)
end

# Patterned after the model.frame function in R
function model_frame(f::Formula, d::AbstractDataFrame)
    ## First extract the names of all variables in the formula
    ## Create the set from the rhs because the lhs can be 'nothing'
    vnms = collect(add_each!(Set(all_vars(f.rhs)...), all_vars(f.lhs)))
    df = d[vnms]
    na_omit(df)
end

model_frame(ex::Expr, d::AbstractDataFrame) = model_frame(Formula(ex), d)

type ModelMatrix{T}
    model::Array{Float64}
    response::Array{Float64}
    model_colnames::Array{T}
    response_colnames::Array{T}
end

# Expand dummy variables and equations
function model_matrix(mf::ModelFrame)
    ex = mf.formula.rhs[1]
    # BUG: complete_cases doesn't preserve grouped columns
    df = mf.df#[complete_cases(mf.df),1:ncol(mf.df)]  
    rdf = df[mf.y_indexes]
    mdf = expand(ex, df)

    # TODO: Convert to Array{Float64} in a cleaner way
    rnames = colnames(rdf)
    mnames = colnames(mdf)
    r = Array(Float64,nrow(rdf),ncol(rdf))
    m = Array(Float64,nrow(mdf),ncol(mdf))
    for i = 1:nrow(rdf)
      for j = 1:ncol(rdf)
        r[i,j] = float(rdf[i,j])
      end
      for j = 1:ncol(mdf)
        m[i,j] = float(mdf[i,j])
      end
    end
    
    include_intercept = true
    if include_intercept
      m = hcat(ones(nrow(mdf)), m)
      unshift!(mnames, "(Intercept)")
    end

    ModelMatrix(m, r, mnames, rnames)
    ## mnames = {}
    ## rnames = {}
    ## for c in 1:ncol(rdf)
    ##   r = hcat(r, float(rdf[c]))
    ##   push!(rnames, colnames(rdf)[c])
    ## end
    ## for c in 1:ncol(mdf)
    ##   m = hcat(m, mdf[c])
    ##   push!(mnames, colnames(mdf)[c])
    ## end
end

model_matrix(f::Formula, d::AbstractDataFrame) = model_matrix(model_frame(f, d))
model_matrix(ex::Expr, d::AbstractDataFrame) = model_matrix(model_frame(Formula(ex), d))

# TODO: Make a more general version of these functions
# TODO: Be able to extract information about each column name
function interaction_design_matrix(a::AbstractDataFrame, b::AbstractDataFrame)
   cols = {}
   col_names = Array(ASCIIString,0)
   for i in 1:ncol(a)
       for j in 1:ncol(b)
          push!(cols, DataArray(a[:,i] .* b[:,j]))
          push!(col_names, string(colnames(a)[i],"&",colnames(b)[j]))
       end
   end
   DataFrame(cols, col_names)
end

function interaction_design_matrix(a::AbstractDataFrame, b::AbstractDataFrame, c::AbstractDataFrame)
   cols = {}
   col_names = Array(ASCIIString,0)
   for i in 1:ncol(a)
       for j in 1:ncol(b)
           for k in 1:ncol(b)
              push!(cols, DataArray(a[:,i] .* b[:,j] .* c[:,k]))
              push!(col_names, string(colnames(a)[i],"&",colnames(b)[j],"&",colnames(c)[k]))
           end
       end
   end
   DataFrame(cols, col_names)
end

# Temporary: Manually describe the interactions needed for DataFrame Array.
function all_interactions(dfs::Array{Any,1})
    d = DataFrame()
    if length(dfs) == 2
      combos = ([1,2],)
    elseif length(dfs) == 3
      combos = ([1,2], [1,3], [2,3], [1,2,3])
    else
      error("interactions with more than 3 terms not implemented (yet)")
    end
    for combo in combos
       if length(combo) == 2
         a = interaction_design_matrix(dfs[combo[1]],dfs[combo[2]])
       elseif length(combo) == 3
         a = interaction_design_matrix(dfs[combo[1]],dfs[combo[2]],dfs[combo[3]])
       end
       d = insert!(d, a)
    end
    return d
end

# string(Expr) now quotes, which we don't want. This hacks around that, stealing
# from print_to_string
function formula_string(ex::Expr)
    s = memio(0, false)
    Base.show_unquoted(s, ex)
    takebuf_string(s)
end

#
# The main expression to DataFrame expansion function.
# Returns a DataFrame.
#

function expand(ex::Expr, df::AbstractDataFrame)
    f = eval(ex.args[1])
    if method_exists(f, (FormulaExpander, Vector{Any}, DataFrame))
        # These are specialized expander functions (+, *, &, etc.)
        f(FormulaExpander(), ex.args[2:end], df)
    else
        # Everything else is called recursively:
        expand(with(df, ex), formula_string(ex), df)
    end
end

function expand(s::Symbol, df::AbstractDataFrame)
    expand(with(df, s), string(s), df)
end

# TODO: make this array{symbol}?
function expand(args::Array{Any}, df::AbstractDataFrame)
    [expand(x, df) for x in args]
end

function expand(x, name::ByteString, df::AbstractDataFrame)
    # If this happens to be a column group, then expand each and concatenate
    if is_group(df, name)
      preds = get_groups(df)[name]
      dfs = [expand(symbol(x), df) for x in preds]
      return cbind(dfs...) 
    end
    # This is the default for expansion: put it right in to a DataFrame.
    DataFrame({x}, [name])
end

#
# Methods for expansion of specific data types
#

# Expand a PooledDataVector into a matrix of indicators for each dummy variable
# TODO: account for NAs?
function expand(poolcol::PooledDataVector, colname::ByteString, df::AbstractDataFrame)
    newcol = {DataArray([convert(Float64,x)::Float64 for x in (poolcol.refs .== i)]) for i in 2:length(poolcol.pool)}
    newcolname = [string(colname, ":", x) for x in poolcol.pool[2:length(poolcol.pool)]]
    DataFrame(newcol, convert(Vector{ByteString}, newcolname))
end


#
# Methods for Formula expansion
#
type FormulaExpander; end # This is an indictor type.

function +(::FormulaExpander, args::Vector{Any}, df::AbstractDataFrame)
    d = DataFrame()
    for a in args
        d = insert!(d, expand(a, df))
    end
    d
end
function (&)(::FormulaExpander, args::Vector{Any}, df::AbstractDataFrame)
    interaction_design_matrix(expand(args[1], df), expand(args[2], df))
end
function *(::FormulaExpander, args::Vector{Any}, df::AbstractDataFrame)
    d = +(FormulaExpander(), args, df)
    d = insert!(d, all_interactions(expand(args, df)))
    d
end
