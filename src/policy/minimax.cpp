#include <utility>
#include "state.hpp"
#include "minimax.hpp"

static void sort_actions(State* state, std::vector<Move>& actions) {
    std::vector<std::pair<int, Move>> scored_moves;
    scored_moves.reserve(actions.size());

    int player = state->player;
    int oppn = 1 - player;

    for (const auto& action : actions) {
        int score = 0;
        Point from = action.first;
        Point to = action.second;

        // Retrieve pieces
        int self_piece = state->piece_at(player, from.first, from.second);
        int oppn_piece = state->piece_at(oppn, to.first, to.second);

        // MVV-LVA for captures
        if (oppn_piece > 0) {
            score += 10000 + (oppn_piece * 100) - self_piece;
        }

        // Pawn promotions
        if (self_piece == 1 && (to.first == 0 || static_cast<int>(to.first) == state->board_h() - 1)) {
            score += 9000;
        }

        // Lightweight center-control / piece advancement heuristic:
        int center_row = state->board_h() / 2;
        int center_col = state->board_w() / 2;
        int from_dist = std::abs((int)from.first - center_row) + std::abs((int)from.second - center_col);
        int to_dist = std::abs((int)to.first - center_row) + std::abs((int)to.second - center_col);
        
        score += (from_dist - to_dist) * 10;

        scored_moves.push_back({score, action});
    }

    for (size_t i = 0; i < scored_moves.size(); i++) {
        size_t best = i;
        for (size_t j = i + 1; j < scored_moves.size(); j++) {
            if (scored_moves[j].first > scored_moves[best].first)
                best = j;
        }
        if (best != i)
            std::swap(scored_moves[i], scored_moves[best]);
    }

    for (size_t i = 0; i < actions.size(); ++i) {
        actions[i] = scored_moves[i].second;
    }
}



/*============================================================
 * MiniMax — eval_ctx
 *
 * Negamax without pruning. Caller manages memory.
 *============================================================*/
int MiniMax::eval_ctx(
    State *state,
    int depth,
    int alpha,
    int beta,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const MMParams& p
){
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }
    if(ctx.stop){
        return 0;
    }

    /* === Lazy move generation (sets game_state) === */
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    /* === Terminal / leaf checks === */

    // [ Hackathon TODO 3-1 ]
    // return the score for a winning terminal state
    // Hint: prefer faster wins by using ply.
    if(state->game_state == WIN){
        return P_MAX - ply;
    }

    if(state->game_state == DRAW){
        return 0;
    }

    /* === Repetition check (game-specific) === */
    int rep_score;
    if(state->check_repetition(history, rep_score)){
        return rep_score;
    }
    history.push(state->hash());

    if(depth <= 0){
        int score = state->evaluate(
            p.use_kp_eval, p.use_eval_mobility, &history
        ); 
        history.pop(state->hash());
        return score;
    }

    if (p.use_move_ordering) {
        sort_actions(state, state->legal_actions);
    }

    /* === Negamax loop === */
    int best_score = M_MAX;
    bool first_move = true;

    for(auto& action : state->legal_actions){
        State* next = static_cast<State*>(state->next_state(action));
        bool same = next->same_player_as_parent();

        int score;
        if (p.use_pvs) {
            if (first_move) {
                int val;
                if (same) {
                    val = eval_ctx(next, depth - 1, alpha, beta, history, ply + 1, ctx, p);
                } else {
                    val = eval_ctx(next, depth - 1, -beta, -alpha, history, ply + 1, ctx, p);
                }
                score = same ? val : -val;
                first_move = false;
            } else {
                // Null window search
                int val;
                if (same) {
                    val = eval_ctx(next, depth - 1, alpha, alpha + 1, history, ply + 1, ctx, p);
                } else {
                    val = eval_ctx(next, depth - 1, -alpha - 1, -alpha, history, ply + 1, ctx, p);
                }
                score = same ? val : -val;

                // Re-search
                if (score > alpha && score < beta) {
                    if (same) {
                        val = eval_ctx(next, depth - 1, score, beta, history, ply + 1, ctx, p);
                    } else {
                        val = eval_ctx(next, depth - 1, -beta, -score, history, ply + 1, ctx, p);
                    }
                    score = same ? val : -val;
                }
            }
        } else {
            // Standard Negamax Alpha-Beta
            int val;
            if (same) {
                val = eval_ctx(next, depth - 1, alpha, beta, history, ply + 1, ctx, p);
            } else {
                val = eval_ctx(next, depth - 1, -beta, -alpha, history, ply + 1, ctx, p);
            }
            score = same ? val : -val;
        }

        delete next;

        if(score > best_score){
            best_score = score;
        }
        if(score > alpha){
            alpha = score;
        }
        if(alpha >= beta){
            break; //prune here
        }
    }

    history.pop(state->hash());
    return best_score;
}


/*============================================================
 * MiniMax — search
 *
 * Iterate legal moves, call eval_ctx, return SearchResult.
 *============================================================*/
SearchResult MiniMax::search(
    State *state,
    int depth,
    GameHistory& history,
    SearchContext& ctx
){
    ctx.reset();
    MMParams p = MMParams::from_map(ctx.params);
    SearchResult result;
    result.depth = depth;

    if(!state->legal_actions.size()){
        state->get_legal_actions();
    }


    if(p.use_move_ordering){
        sort_actions(state, state->legal_actions);
    }

    int alpha = M_MAX - 10;
    int beta = P_MAX + 10;
    int best_score = M_MAX - 10;
    int move_index = 0;
    int total_moves = (int)state->legal_actions.size();
    bool first_move = true;

    for(auto& action : state->legal_actions){
        State* next = static_cast<State*>(state->next_state(action));
        bool same = next->same_player_as_parent();
        
        int score;
        if(p.use_pvs){
            if(first_move){
                int val;
                if(same) {
                    val = eval_ctx(next, depth - 1, alpha, beta, history, 1, ctx, p);
                } else {
                    val = eval_ctx(next, depth - 1, -beta, -alpha, history, 1, ctx, p);
                }
                score = same ? val : -val;
                first_move = false;
            } else {
                // Null window search
                int val;
                if(same) {
                    val = eval_ctx(next, depth - 1, alpha, alpha + 1, history, 1, ctx, p);
                } else {
                    val = eval_ctx(next, depth - 1, -alpha - 1, -alpha, history, 1, ctx, p);
                }
                score = same ? val : -val;

                // Re-search
                if(score > alpha && score < beta){
                    if(same) {
                        val = eval_ctx(next, depth - 1, score, beta, history, 1, ctx, p);
                    } else {
                        val = eval_ctx(next, depth - 1, -beta, -score, history, 1, ctx, p);
                    }
                    score = same ? val : -val;
                }
            }
        } else {
            // Standard Negamax
            int val;
            if(same) {
                val = eval_ctx(next, depth - 1, alpha, beta, history, 1, ctx, p);
            } else {
                val = eval_ctx(next, depth - 1, -beta, -alpha, history, 1, ctx, p);
            }
            score = same ? val : -val;
        }
        delete next;

        if(score > best_score){
            best_score = score;
            result.best_move = action;

            if(p.report_partial && ctx.on_root_update){
               ctx.on_root_update({result.best_move, best_score, depth, move_index + 1, total_moves});
            }
        }
        if(score > alpha){
            alpha = score;
        }
        move_index++;
    }

    // [ Hackathon TODO 4-3 ]
    // update result and return
    result.score = best_score;
    result.nodes = ctx.nodes;
    result.seldepth = ctx.seldepth;
    result.pv = {result.best_move};

    return result;
} 


/*============================================================
 * MiniMax — default_params / param_defs
 *============================================================*/
ParamMap MiniMax::default_params(){
    return {
        {"UseKPEval", "true"},
        {"UseEvalMobility", "true"},
        {"ReportPartial", "true"},
        {"UsePVS", "true"},
        {"UseMoveOrdering", "true"},
    };
}

std::vector<ParamDef> MiniMax::param_defs(){
    return {
        {"UseKPEval", ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial", ParamDef::CHECK, "true"},
        {"UsePVS", ParamDef::CHECK, "true"},
        {"UseMoveOrdering", ParamDef::CHECK, "true"},
    };
}
