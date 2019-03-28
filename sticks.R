#https://github.com/aboulaboul
#translate from https://github.com/thibo73800 Python script
StickGame <- setRefClass("StickGame",
                         fields = list(nb = "numeric",
                                       original_nb = "numeric"),
                         methods = list(
                           initialize = function(x=12){
                             nb <<- x
                             original_nb <<- nb
                           },
                           is_finished = function(){
                             if (nb <= 0) {return(TRUE)} else {return(FALSE)}
                           },
                           reset = function(){
                             nb <<- original_nb
                           },
                           display = function(){
                             cat(paste(rep('|', nb), collapse = " "))
                           },
                           step = function(action){
                             # @action either 1, 2 or 3. Take an action into the environement
                             nb <<- nb - action
                             #r
                             if (nb <= 0)
                             {
                               sp = 0; r = -1
                             } else {
                               sp = nb; r = 0
                             }
                             return(list(sp = sp, r = r))
                           })
)

StickPlayer <- setRefClass("StickPlayer",
                           fields = list(is_human = "numeric",
                                         size = "numeric",
                                         trainable = "numeric",
                                         history = "data.frame",
                                         V = "numeric",
                                         win_nb = "numeric",
                                         lose_nb = "numeric",
                                         rewards = "numeric",
                                         eps = "numeric"
                           ),
                           methods = list(
                             initialize = function(is_human, size, trainable = TRUE){
                               is_human <<- is_human
                               history <<- data.frame(s = 0, a = 0, r = 0, sp = 0)
                               V <<- rep(0, size)
                               win_nb <<- 0
                               lose_nb <<- 0
                               rewards <<- 0
                               eps <<- 0.99
                               trainable <<- trainable
                             },
                             reset_stat = function(){
                               win_nb <<- 0
                               lose_nb <<- 0
                               rewards <<- 0
                             },
                             greedy_step = function(state){
                               actions = c(1, 2, 3)
                               vmin = Inf
                               vi = NULL
                               for (i in 1:length(actions))
                               {
                                 a = actions[i]
                                 vi = 1
                                 if ((state - a) > 0)
                                 {
                                   if (is.infinite(vmin) | vmin > V[state - a])
                                   {
                                     vmin = V[state - a]
                                     vi = i
                                   }
                                 }
                               }
                               return(actions[vi])
                             },
                             play = function(state){
                               # play given the @state (int)
                               if (!is_human)
                               {
                                 #take random action
                                 if (runif(n = 1, min = 0, max = 1) < eps)
                                 {
                                   action = sample.int(n = 3, size = 1)
                                 } else {
                                   action = greedy_step(state)
                                 }
                               } else {
                                 action = as.numeric(readline(prompt = "number of sticks to take:"))
                               }
                               return(action)
                             },
                             add_transition = function(n_tuple){
                               # Add one transition to the history: tuple (s, a , r, s')
                               history <<- rbind(history, n_tuple)
                               s <- n_tuple[1];a <- n_tuple[2]; r <- n_tuple[3]; sp <- n_tuple[4]
                               rewards[length(rewards) + 1] <<- r
                             },
                             train = function(){
                               if (trainable == 1 & is_human == 0)
                               {
                                 # Update the value function if this player is not human
                                 for (transition in nrow(history):1)
                                 {
                                   n_tuple = as.numeric(history[transition,])
                                   s <- n_tuple[1];a <- n_tuple[2]; r <- n_tuple[3]; sp <- n_tuple[4]
                                   if (r == 0)
                                   {
                                     V[s] <<- V[s] + 0.01 * (V[sp] - V[s])
                                   } else {
                                     V[s] <<- V[s] + 0.01 * (r - V[s])
                                   }
                                 }
                               }
                               history <<- data.frame(s = 0, a = 0, r = 0, sp = 0)
                             }
                           )
)

play <- function(game, p1, p2, train = TRUE)
{
  game$reset()
  state = game$nb
  players = list(p1, p2)
  playersid = sample(x = length(players), size = length(players), replace = FALSE)
  p = 1; op = 2
  while (!game$is_finished())
  {
    if (players[[playersid[p]]]$is_human) game$display()

    action <- players[[playersid[p]]]$play(state)
    sp_r <- game$step(action)
    n_state <- sp_r$sp
    reward <- sp_r$r

    #game is over. Ass stat
    if (reward != 0)
    {
      # Update stat of the current player
      players[[playersid[p]]]$lose_nb <- players[[playersid[p]]]$lose_nb + 1
      # Update stat of the other player
      players[[playersid[op]]]$win_nb <- players[[playersid[op]]]$win_nb + 1
    }

    # Add the reversed reward and the new state to the other player
    players[[playersid[p]]]$add_transition(c(state, action, reward, NA))
    players[[playersid[op]]]$history[nrow(players[[playersid[op]]]$history), c("r","sp")] <- c(-reward, n_state)

    state = n_state
    op <- p
    p  <- ifelse(p == 1, 2, 1)
  }
  if (train == TRUE)
  {
    p1$train()
    p2$train()
  }
}


game <- StickGame()
p1 <- StickPlayer(is_human = 0, size = 12, trainable = 1)
p2 <- StickPlayer(is_human = 0, size = 12, trainable = 1)
human <- StickPlayer(is_human = 1, size = 12, trainable = 0)
random_player <- StickPlayer(is_human = 0, size = 12, trainable = 0)



#play(game, p1, p2, train = TRUE)

# Train the agents
for (i in 1:10000)
{
  if (i %% 10 == 0)
  {
    p1$eps = max(p1$eps*0.996, 0.05)
    p2$eps = max(p2$eps*0.996, 0.05)
    #if (i %% 100 == 0) cat(paste('p1', p1$win_nb, 'p2', p2$win_nb, '\n'))
  }
  play(game, p1, p2, train = TRUE)
}
# Display the value function
#p1$V
#p2$V

# Play agains a random player
p1$eps = p2$eps <- .05
for (i in 1:100)
{
  play(game, p2, random_player, train = FALSE)
  play(game, p1, random_player, train = FALSE)
}
cat(paste('p1 wins: ', p1$win_nb, 'defeats: ', p1$lose_nb, '\n'))
cat(paste('p2 wins: ', p2$win_nb, 'defeats: ', p2$lose_nb, '\n'))

# Play against us
p1$eps = .05
play(game, p1, human, train=FALSE)
