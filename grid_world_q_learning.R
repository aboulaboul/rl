#https://github.com/aboulaboul
#translate from https://github.com/thibo73800 Python script
EnvGrid <- setRefClass("EnvGrid",
                       fields = list(mygrid = "matrix",
                                     actions = "matrix",
                                     y = "numeric",
                                     x = "numeric"),
                       methods = list(
                         initialize = function(){
                           mygrid <<- matrix(data = c(0,0,1, 0,-1,0, 0,0,0),
                                             nrow = 3, ncol = 3,
                                             byrow = TRUE)
                           #start position
                           y <<- 3
                           x <<- 1

                           actions <<- matrix(data = c(-1,0,
                                                       1,0,
                                                       0,-1,
                                                       0,1),
                                              nrow = 4, ncol = 2,
                                              byrow = TRUE);#up,down, left,right

                         },
                         reset = function(){
                           #Reset world
                           y <<- 3
                           x <<- 1
                           return((y - 1) * 3 + x)
                         },
                         mystep = function(action){
                           #Action: 0, 1, 2, 3
                           y <<- max(1, min(y + actions[action, 1], 3))
                           x <<- max(1, min(x + actions[action, 2], 3))
                           return(list((y - 1) * 3 + x, mygrid[y, x]))
                         },
                         myshow = function(){
                           #Show the grid
                           cat(paste0(paste(rep('-',20), collapse = ''), "\n"))
                           for (myline in 1:dim(mygrid)[1])
                           {
                             xdisp = 0
                             for (mycol in 1:dim(mygrid)[2])
                             {
                               if (mycol != x | myline != y) cat(mygrid[myline, mycol]) else cat("X")
                               cat("\t")
                             }
                             cat("\n")
                           }
                         },
                         is_finished = function(){
                           return(mygrid[y,x] == 1)
                         })
)

take_action <- function(st, Q, eps)
{
  # Take an action
  if (runif(n = 1, min = 0, max = 1) < eps)
  {
    action = sample(4, size = 1)
  } else {
    action = which.max(Q[st,])
  }
  return(action)
}


env <- EnvGrid()
st <- env$reset()

Q <- matrix(data = rep(0, 36), nrow = 9, ncol = 4)

for (i in 1:1000)
{
  # Reset the game
  st <- env$reset()
  while (!env$is_finished())
  {
    #env$myshow()
    at <- take_action(st, Q, 0.4)
    myprovres <- env$mystep(at)
    stp1 <- myprovres[[1]]
    r <- myprovres[[2]]
    cat(paste("s", stp1, ' _ '))
    cat(paste("r", r, '\n'))
    # Update Q function
    atp1 <- take_action(stp1, Q, 0.0)
    Q[st, at] = Q[st, at] + 0.1*(r + 0.9*Q[stp1, atp1] - Q[st, at])

    st = stp1
  }

}
