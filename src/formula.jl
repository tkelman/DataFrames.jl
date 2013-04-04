# Formulas for representing and working with linear-model-type expressions
# Original by Harlan D. Harris.  Later modifications by John Myles White
# and Douglas M. Bates.

## Formulas are written as expressions and parsed by the Julia parser.
## For example :(y ~ a + b + log(c))
## In Julia the & operator is used for an interaction.  What would be written
## in R as y ~ a + b + a:b is written :(y ~ a + b + a&b) in Julia.

## Each side of a formula is either a Symbol or an Expr

typealias SymExpr Union(Symbol,Expr)

## The lhs of a one-sided formula is 'nothing'

type Formula
    lhs::Union(SymExpr,Nothing)
    rhs::SymExpr
end

function Formula(ex::Expr) 
    aa = ex.args
    if aa[1] != :~
        error("Invalid formula, top-level argument must be '~'.  Check parentheses.")
    end
    if length(aa) == 2 return Formula(nothing, aa[2]) end
    Formula(aa[2], aa[3])
end

function show(io::IO, f::Formula)
    print(io, string("Formula: ", f.lhs == nothing ? "" : f.lhs, " ~ ", f.rhs))
end

type Terms
    terms::Vector
    variables::Vector{Symbol}
    factors::Matrix{Int8}               # variables to terms
    order::Vector{Int8}
    response::Int
    intercept::Bool
end

type ModelFrame
    df::AbstractDataFrame
    terms::Terms
    msng::BitArray
end

## Return, as a vector of symbols, the names of all the variables in
## an expression or a formula
function allvars(ex::Expr)
    [[allvars(a) for a in ex.args[2:end]]...]
end
function allvars(f::Formula)
    ## Create the set from the rhs then add_each for the lhs, which can be 'nothing'
    collect(add_each!(Set(allvars(f.rhs)...), allvars(f.lhs)))
end
allvars(sym::Symbol) = [sym]
allvars(symv::Vector{Symbol}) = symv
allvars(v) = Array(Symbol,0)

function getterms(ex::Expr)
    ex.args[1] == :+ ? ex.args[2:end] : [ex]
end
getterms(f::Formula) = getterms(f.rhs)
getterms(sym::Symbol) = [sym]

function expandasterisk(ex::Expr)
    if ex.args[1] != :* return ex end
    template = :(u&v)
    a2 = ex.args[2]
    a3 = expandasterisk(ex.args[3])
    template.args[2] = a2
    template.args[3] = a3
    if isa(a3, Symbol) return [a2, a3, template] end
    vcat([a2, a3, [:(a2&a) for a in a3]])
end
expandasterisk(v::Vector) = vcat([expandasterisk(vv) for vv in v])
expandasterisk(s::Symbol) = s

## Extract the terms and orders from a formula
## ToDo: Expand a*b, a/b, etc. on the rhs
##       Handle cases where rhs top level operator is - or any toplevel
##       arg is a subtraction 
function Terms(f::Formula, d::AbstractDataFrame)
    vars = allvars(f)
    terms = getterms(f)
    terms = terms[!(terms .== 1)]          # drop any explicit 1's
    noint = (terms .== 0) | (terms .== -1) # should also handle :(-(expr,1))
    terms = terms[!noint]
    ## expand terms of the form a*b to a + b + (a&b)
    terms = expandasterisk(terms)
    ## create the boolean array mapping factors to terms
    factors = hcat(map(t->(vv = Set(allvars(t)...);
                           convert(Vector{Int8},map(x->has(vv,x),vars))),
                       terms)...)
    ord = vec(sum(factors,[1]))
    if !issorted(ord)
        pp = sortperm(ord)
        terms = terms[pp]
        factors = factors[:,pp]
    end
    response = 0
    if f.lhs != nothing
        terms = unshift!(terms, f.lhs)
        response = 1
    end
    Terms(terms, vars, factors, ord, response, !any(noint))
end

## Default NA handler.  Others can be added when keyword arguments are available.
function na_omit(df::DataFrame)
    msng = vec(reducedim(|, isna(df), [2], false))
    df[!msng,:], msng
end

function dropUnusedLevels!(da::PooledDataArray)
    rr = da.refs
    uu = unique(rr)
    if length(uu) == length(da.pool) return da end
    T = eltype(rr)
    su = sort!(uu)
    dict = Dict(su, one(T):convert(T,length(uu)))
    da.refs = map(x->dict[x], rr)
    da.pool = da.pool[uu]
    da
end
dropUnusedLevels!(x) = x
    
function ModelFrame(f::Formula, d::AbstractDataFrame)
    trms = Terms(f, d)
    ## Select only the variables from the formula and apply the NA handler
    df = d[trms.variables]
    df, msng = na_omit(df)
    for i in 1:length(df) df[i] = dropUnusedLevels!(df[i]) end
#    dd = DataFrame()
#    for t in trms.terms dd[string(t)] = with(df, t) end
    ModelFrame(df, trms, msng)
end
ModelFrame(ex::Expr, d::AbstractDataFrame) = ModelFrame(Formula(ex), d)

model_response(mf::ModelFrame) = with(mf.df, mf.terms.terms[mf.terms.response])

function contr_treatment(n::Integer, contrasts::Bool, sparse::Bool, base::Integer)
    if n < 2 error("not enought degrees of freedom to define contrasts") end
    contr = sparse ? speye(n) : eye(n)
    if !contrasts return contr end
    if !(1 <= base <= n) error("base = $base is not allowed for n = $n") end
    contr[:,vcat(1:(base-1),(base+1):end)]
end
contr_treatment(n::Integer,contrasts::Bool,sparse::Bool) = contr_treatment(n,contrasts,sparse,1)
contr_treatment(n::Integer, contrasts::Bool) = contr_treatment(n,contrasts,false,1)
contr_treatment(n::Integer) = contr_treatment(n,true,false,1)
    
type ModelMatrix
    m::Matrix{Float64}
    assign::Vector{Int}
#    contrasts::Vector{Function}
end

function cols(vv)
    vtyp = typeof(vv)
    if vtyp <: PooledDataVector
        return contr_treatment(length(vv.pool))[vv.refs,:]
    end
    if !(vtyp <: DataVector)
        error("column generator is neither a PooledDataVector nor a DataVector")
    end
    float64(vv.data)
end

function ModelMatrix(mf::ModelFrame)
    trms = mf.terms
    cdict = Dict(trms.variables, [cols(mf.df[v]) for v in trms.variables])
    vv = [cdict[nm] for nm in trms.terms]
    if trms.intercept unshift!(vv, ones(size(mf.df,1))) end
    mm = hcat(vv...)
    ModelMatrix(mm, ones(Int, size(mm,2)))
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
