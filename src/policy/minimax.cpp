#include <utility>
#include <vector>
#include <algorithm>
#include "state.hpp"
#include "minimax.hpp"

enum TTBound : uint8_t {
    TT_EXACT = 0,
    TT_ALPHA = 1,
    TT_BETA = 2
};

struct TTEntry {
    uint64_t hash = 0;
    int score = 0;
    int depth = -1;
    uint8_t bound = 0;
    Move best_move = Move();
};

class TranspositionTable {
private:
    std::vector<TTEntry> table;
public:
    TranspositionTable(size_t size = 1 << 20) : table(size) {}

    void clear() {
        std::fill(table.begin(), table.end(), TTEntry{0, 0, -1, 0, Move()});
    }

    TTEntry* lookup(uint64_t hash) {
        size_t idx = hash & (table.size() - 1);
        if (table[idx].hash == hash) {
            return &table[idx];
        }
        return nullptr;
    }

    void store(uint64_t hash, int score, int depth, uint8_t bound, const Move& best_move) {
        size_t idx = hash & (table.size() - 1);
        if (table[idx].hash == 0 || depth >= table[idx].depth) {
            table[idx] = {hash, score, depth, bound, best_move};
        }
    }
};

static TranspositionTable tt;

static Move killer_moves[2][128];

static void clear_killers() {
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 128; j++) {
            killer_moves[i][j] = Move();
        }
    }
}

inline int score_to_tt(int score, int ply) {
    if (score > 90000) return score + ply;
    if (score < -90000) return score - ply;
    return score;
}

inline int score_from_tt(int score, int ply) {
    if (score > 90000) return score - ply;
    if (score < -90000) return score + ply;
    return score;
}

static bool has_major_pieces(State *state) {
    int p = state->player;
    for (int r = 0; r < BOARD_H; r++) {
        for (int c = 0; c < BOARD_W; c++) {
            int piece = state->piece_at(p, r, c);
            if (piece > 1 && piece < 6) {
                return true;
            }
        }
    }
    return false;
}

static void sort_actions(
    State *state,
    std::vector<Move> &actions,
    const Move& hash_move = Move(),
    const Move& killer1 = Move(),
    const Move& killer2 = Move())
{
    std::vector<std::pair<int, Move>> scored_moves;
    scored_moves.reserve(actions.size());

    int player = state->player;
    int oppn = 1 - player;

    for (const auto &action : actions)
    {
        int score = 0;
        Point from = action.first;
        Point to = action.second;

        // Hash move gets absolute top priority
        if (action == hash_move)
        {
            score += 1000000;
        }

        // Retrieve pieces
        int self_piece = state->piece_at(player, from.first, from.second);
        int oppn_piece = state->piece_at(oppn, to.first, to.second);

        // MVV-LVA for captures
        if (oppn_piece > 0)
        {
            score += 10000 + (oppn_piece * 100) - self_piece;
        }
        // Pawn promotions
        else if (self_piece == 1 && (to.first == 0 || static_cast<int>(to.first) == state->board_h() - 1))
        {
            score += 9000;
        }
        // Killer move 1
        else if (action == killer1)
        {
            score += 8000;
        }
        // Killer move 2
        else if (action == killer2)
        {
            score += 7000;
        }

        // Lightweight center-control / piece advancement heuristic:
        int center_row = state->board_h() / 2;
        int center_col = state->board_w() / 2;
        int from_dist = std::abs((int)from.first - center_row) + std::abs((int)from.second - center_col);
        int to_dist = std::abs((int)to.first - center_row) + std::abs((int)to.second - center_col);

        score += (from_dist - to_dist) * 10;

        scored_moves.push_back({score, action});
    }

    for (size_t i = 0; i < scored_moves.size(); i++)
    {
        size_t best = i;
        for (size_t j = i + 1; j < scored_moves.size(); j++)
        {
            if (scored_moves[j].first > scored_moves[best].first)
                best = j;
        }
        if (best != i)
            std::swap(scored_moves[i], scored_moves[best]);
    }

    for (size_t i = 0; i < actions.size(); ++i)
    {
        actions[i] = scored_moves[i].second;
    }
}

// ================= Quiescence thingy
int MiniMax::qsearch(
    State *state,
    int alpha,
    int beta,
    GameHistory &history,
    int ply,
    SearchContext &ctx,
    const MMParams &p,
    int q_depth)
{
    ctx.nodes++;
    if (ctx.stop)
        return 0;

    // Hard stop — too deep, just evaluate
    if (q_depth <= 0)
        return state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history);

    // Terminal check
    if (state->legal_actions.empty() && state->game_state == UNKNOWN)
    {
        state->get_legal_actions();
    }
    if (state->game_state == WIN)
        return P_MAX - ply;
    if (state->game_state == DRAW)
        return 0;
    if (state->legal_actions.empty())
        return M_MAX + ply;

    if (p.use_move_ordering)
    {
        sort_actions(state, state->legal_actions);
    }

    // Standing pat — evaluate current position without any capture
    int stand_pat = state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history);

    // If standing pat already beats beta, prune immediately
    if (stand_pat >= beta)
        return stand_pat;

    // Standing pat raises alpha floor
    if (stand_pat > alpha)
        alpha = stand_pat;

    // Only look at capture moves
    for (auto &action : state->legal_actions)
    {
        int oppn = 1 - state->player;
        Point to = action.second;
        int victim = state->piece_at(oppn, to.first, to.second);

        // Skip non-captures
        if (victim == 0)
            continue;

        State *next = static_cast<State *>(state->next_state(action));
        bool same = next->same_player_as_parent();

        int val;
        if (same)
        {
            val = qsearch(next, alpha, beta, history, ply + 1, ctx, p, q_depth - 1);
        }
        else
        {
            val = qsearch(next, -beta, -alpha, history, ply + 1, ctx, p, q_depth - 1);
        }
        int score = same ? val : -val;
        delete next;

        if (score >= beta)
            return score; // beta cutoff
        if (score > alpha)
            alpha = score;
    }

    return alpha;
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
    GameHistory &history,
    int ply,
    SearchContext &ctx,
    const MMParams &p,
    bool allowed_null)
{
    ctx.nodes++;
    if (ply > ctx.seldepth)
    {
        ctx.seldepth = ply;
    }
    if (ctx.stop)
    {
        return 0;
    }

    /* === Lazy move generation (sets game_state) === */
    if (state->legal_actions.empty() && state->game_state == UNKNOWN)
    {
        state->get_legal_actions();
    }

    /* === Terminal / leaf checks === */

    // [ Hackathon TODO 3-1 ]
    // return the score for a winning terminal state
    // Hint: prefer faster wins by using ply.
    if (state->game_state == WIN)
    {
        return P_MAX - ply;
    }

    if (state->game_state == DRAW)
    {
        return 0;
    }

    /* === Repetition check (game-specific) === */
    int rep_score;
    if (state->check_repetition(history, rep_score))
    {
        return rep_score;
    }

    // TT Lookup
    uint64_t hash = state->hash();
    TTEntry* entry = nullptr;
    Move hash_move = Move();
    if (p.use_tt) {
        entry = tt.lookup(hash);
        if (entry) {
            hash_move = entry->best_move;
            if (entry->depth >= depth) {
                int tt_score = score_from_tt(entry->score, ply);
                if (entry->bound == TT_EXACT) {
                    return tt_score;
                }
                if (entry->bound == TT_ALPHA && tt_score <= alpha) {
                    return tt_score;
                }
                if (entry->bound == TT_BETA && tt_score >= beta) {
                    return tt_score;
                }
            }
        }
    }

    // NMP (Null Move Pruning)
    if (p.use_nmp && allowed_null && depth >= 3 && has_major_pieces(state)) {
        int static_val = state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history);
        if (static_val >= beta) {
            State* null_state = static_cast<State*>(state->create_null_state());
            if (null_state) {
                int R = 2;
                int null_depth = depth - 1 - R;
                if (null_depth < 0) null_depth = 0;

                int val = eval_ctx(null_state, null_depth, -beta, -beta + 1, history, ply + 1, ctx, p, false);
                int score = -val;
                delete null_state;

                if (score >= beta) {
                    return beta; // Prune!
                }
            }
        }
    }

    history.push(hash);

    if (depth <= 0)
    {
        if (p.use_quiescence)
        {
            // NEW: run qsearch instead of plain evaluate
            int score = qsearch(state, alpha, beta, history, ply, ctx, p);
            history.pop(hash);
            return score;
        }
        int score = state->evaluate(
            p.use_kp_eval, p.use_eval_mobility, &history);
        history.pop(hash);
        return score;
    }

    if (p.use_move_ordering)
    {
        if (p.use_killers && ply < 128) {
            sort_actions(state, state->legal_actions, hash_move, killer_moves[0][ply], killer_moves[1][ply]);
        } else {
            sort_actions(state, state->legal_actions, hash_move);
        }
    }

    /* === Negamax loop === */
    int best_score = M_MAX;
    Move best_move = Move();
    int alpha_orig = alpha;
    bool first_move = true;
    int move_count = 0;

    for (auto &action : state->legal_actions)
    {
        State *next = static_cast<State *>(state->next_state(action));
        bool same = next->same_player_as_parent();
        move_count++;

        int oppn = 1 - state->player;
        Point to = action.second;
        int victim = state->piece_at(oppn, to.first, to.second);
        int self_piece = state->piece_at(state->player, action.first.first, action.first.second);
        bool is_promo = (self_piece == 1 && (to.first == 0 || static_cast<int>(to.first) == state->board_h() - 1));

        int score;
        bool do_lmr = false;

        if (p.use_lmr && depth >= 3 && move_count > 3 && victim == 0 && !is_promo && action != hash_move && action != killer_moves[0][ply] && action != killer_moves[1][ply]) {
            do_lmr = true;
        }

        if (do_lmr) {
            int reduction = 1;
            if (depth >= 6) reduction = 2;
            int reduced_depth = depth - 1 - reduction;
            if (reduced_depth < 0) reduced_depth = 0;

            int val;
            if (same) {
                val = eval_ctx(next, reduced_depth, alpha, alpha + 1, history, ply + 1, ctx, p);
            } else {
                val = eval_ctx(next, reduced_depth, -alpha - 1, -alpha, history, ply + 1, ctx, p);
            }
            score = same ? val : -val;

            if (score > alpha) {
                if (same) {
                    val = eval_ctx(next, depth - 1, alpha, beta, history, ply + 1, ctx, p);
                } else {
                    val = eval_ctx(next, depth - 1, -beta, -alpha, history, ply + 1, ctx, p);
                }
                score = same ? val : -val;
            }
        }
        else
        {
            if (p.use_pvs)
            {
                if (first_move)
                {
                    int val;
                    if (same)
                    {
                        val = eval_ctx(next, depth - 1, alpha, beta, history, ply + 1, ctx, p);
                    }
                    else
                    {
                        val = eval_ctx(next, depth - 1, -beta, -alpha, history, ply + 1, ctx, p);
                    }
                    score = same ? val : -val;
                    first_move = false;
                }
                else
                {
                    // Null window search
                    int val;
                    if (same)
                    {
                        val = eval_ctx(next, depth - 1, alpha, alpha + 1, history, ply + 1, ctx, p);
                    }
                    else
                    {
                        val = eval_ctx(next, depth - 1, -alpha - 1, -alpha, history, ply + 1, ctx, p);
                    }
                    score = same ? val : -val;

                    // Re-search
                    if (score > alpha && score < beta)
                    {
                        if (same)
                        {
                            val = eval_ctx(next, depth - 1, score, beta, history, ply + 1, ctx, p);
                        }
                        else
                        {
                            val = eval_ctx(next, depth - 1, -beta, -score, history, ply + 1, ctx, p);
                        }
                        score = same ? val : -val;
                    }
                }
            }
            else
            {
                // Standard Negamax Alpha-Beta
                int val;
                if (same)
                {
                    val = eval_ctx(next, depth - 1, alpha, beta, history, ply + 1, ctx, p);
                }
                else
                {
                    val = eval_ctx(next, depth - 1, -beta, -alpha, history, ply + 1, ctx, p);
                }
                score = same ? val : -val;
            }
        }

        delete next;
        first_move = false;

        if (score > best_score)
        {
            best_score = score;
            best_move = action;
        }
        if (score > alpha)
        {
            alpha = score;
        }
        if (alpha >= beta)
        {
            if (p.use_killers && ply < 128) {
                int oppn = 1 - state->player;
                Point to = action.second;
                int victim = state->piece_at(oppn, to.first, to.second);
                if (victim == 0) { // quiet move caused beta cutoff
                    if (action != killer_moves[0][ply]) {
                        killer_moves[1][ply] = killer_moves[0][ply];
                        killer_moves[0][ply] = action;
                    }
                }
            }
            break; // prune here
        }
    }

    if (p.use_tt && !ctx.stop) {
        uint8_t bound = TT_EXACT;
        if (best_score <= alpha_orig) {
            bound = TT_ALPHA;
        } else if (best_score >= beta) {
            bound = TT_BETA;
        }
        tt.store(hash, score_to_tt(best_score, ply), depth, bound, best_move);
    }

    history.pop(hash);
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
    GameHistory &history,
    SearchContext &ctx)
{
    ctx.reset();
    MMParams p = MMParams::from_map(ctx.params);
    SearchResult result;
    result.depth = depth;

    // Detect new game or reset to clear the TT and killers
    if (depth == 1) {
        static int last_step = -1;
        if (state->step == 0 || state->step < last_step) {
            tt.clear();
            clear_killers();
        }
        last_step = state->step;
    }

    if (!state->legal_actions.size())
    {
        state->get_legal_actions();
    }

    uint64_t hash = state->hash();
    Move hash_move = Move();
    if (p.use_tt) {
        TTEntry* entry = tt.lookup(hash);
        if (entry) {
            hash_move = entry->best_move;
        }
    }

    if (p.use_move_ordering)
    {
        sort_actions(state, state->legal_actions, hash_move);
    }

    int alpha = M_MAX - 10;
    int beta = P_MAX + 10;
    int best_score = M_MAX - 10;
    int move_index = 0;
    int total_moves = (int)state->legal_actions.size();
    bool first_move = true;

    for (auto &action : state->legal_actions)
    {
        State *next = static_cast<State *>(state->next_state(action));
        bool same = next->same_player_as_parent();

        int score;
        if (p.use_pvs)
        {
            if (first_move)
            {
                int val;
                if (same)
                {
                    val = eval_ctx(next, depth - 1, alpha, beta, history, 1, ctx, p);
                }
                else
                {
                    val = eval_ctx(next, depth - 1, -beta, -alpha, history, 1, ctx, p);
                }
                score = same ? val : -val;
                first_move = false;
            }
            else
            {
                // Null window search
                int val;
                if (same)
                {
                    val = eval_ctx(next, depth - 1, alpha, alpha + 1, history, 1, ctx, p);
                }
                else
                {
                    val = eval_ctx(next, depth - 1, -alpha - 1, -alpha, history, 1, ctx, p);
                }
                score = same ? val : -val;

                // Re-search
                if (score > alpha && score < beta)
                {
                    if (same)
                    {
                        val = eval_ctx(next, depth - 1, score, beta, history, 1, ctx, p);
                    }
                    else
                    {
                        val = eval_ctx(next, depth - 1, -beta, -score, history, 1, ctx, p);
                    }
                    score = same ? val : -val;
                }
            }
        }
        else
        {
            // Standard Negamax
            int val;
            if (same)
            {
                val = eval_ctx(next, depth - 1, alpha, beta, history, 1, ctx, p);
            }
            else
            {
                val = eval_ctx(next, depth - 1, -beta, -alpha, history, 1, ctx, p);
            }
            score = same ? val : -val;
        }
        delete next;

        if (score > best_score)
        {
            best_score = score;
            result.best_move = action;

            if (p.report_partial && ctx.on_root_update)
            {
                ctx.on_root_update({result.best_move, best_score, depth, move_index + 1, total_moves});
            }
        }
        if (score > alpha)
        {
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

    if (p.use_tt && !ctx.stop) {
        uint8_t bound = TT_EXACT;
        if (best_score <= M_MAX - 10) {
            bound = TT_ALPHA;
        }
        tt.store(hash, score_to_tt(best_score, 0), depth, bound, result.best_move);
    }

    return result;
}

/*============================================================
 * MiniMax — default_params / param_defs
 *============================================================*/
ParamMap MiniMax::default_params()
{
    return {
        {"UseKPEval", "true"},
        {"UseEvalMobility", "false"},
        {"ReportPartial", "true"},
        {"UsePVS", "true"},
        {"UseMoveOrdering", "true"},
        {"UseQuiescence", "true"},
        {"UseTT", "true"},
        {"UseNMP", "true"},
        {"UseKillerMoves", "true"},
        {"UseLMR", "true"},
    };
}

std::vector<ParamDef> MiniMax::param_defs()
{
    return {
        {"UseKPEval", ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "false"},
        {"ReportPartial", ParamDef::CHECK, "true"},
        {"UsePVS", ParamDef::CHECK, "true"},
        {"UseMoveOrdering", ParamDef::CHECK, "true"},
        {"UseQuiescence", ParamDef::CHECK, "true"},
        {"UseTT", ParamDef::CHECK, "true"},
        {"UseNMP", ParamDef::CHECK, "true"},
        {"UseKillerMoves", ParamDef::CHECK, "true"},
        {"UseLMR", ParamDef::CHECK, "true"},
    };
}
