#pragma once
#include "search_types.hpp"
#include "game_history.hpp"

struct MMParams
{
    bool use_kp_eval = true;
    bool use_eval_mobility = false;
    bool report_partial = true;
    bool use_pvs = true;
    bool use_move_ordering = true;
    bool use_quiescence = true;
    bool use_tt = true;
    bool use_nmp = true;
    bool use_killers = true;
    bool use_lmr = true;

    static MMParams from_map(const ParamMap &m)
    {
        MMParams p;
        p.use_kp_eval = param_bool(m, "UseKPEval", true);
        p.use_eval_mobility = param_bool(m, "UseEvalMobility", false);
        p.report_partial = param_bool(m, "ReportPartial", true);
        p.use_pvs = param_bool(m, "UsePVS", true);
        p.use_move_ordering = param_bool(m, "UseMoveOrdering", true);
        p.use_quiescence = param_bool(m, "UseQuiescence", true);
        p.use_tt = param_bool(m, "UseTT", true);
        p.use_nmp = param_bool(m, "UseNMP", true);
        p.use_killers = param_bool(m, "UseKillerMoves", true);
        p.use_lmr = param_bool(m, "UseLMR", true);
        return p;
    }
};

class MiniMax
{
public:
    static int qsearch( // NEW
        State *state,
        int alpha,
        int beta,
        GameHistory &history,
        int ply,
        SearchContext &ctx,
        const MMParams &p,
        int q_depth = 6);

    static int eval_ctx(
        State *state,
        int depth,
        int alpha,
        int beta,
        GameHistory &history,
        int ply,
        SearchContext &ctx,
        const MMParams &p,
        bool allowed_null = true);
    static SearchResult search(
        State *state,
        int depth,
        GameHistory &history,
        SearchContext &ctx);

    static ParamMap default_params();
    static std::vector<ParamDef> param_defs();
};
