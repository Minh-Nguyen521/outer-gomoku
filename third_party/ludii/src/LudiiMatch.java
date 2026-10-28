import compiler.Compiler;
import main.grammar.Description;
import main.options.UserSelections;
import main.grammar.Report;
import game.Game;
import other.AI;
import other.context.Context;
import other.move.Move;
import other.trial.Trial;
import utils.RandomAI;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Headless harness: plays Outer-Open Gomoku games in Ludii between two
 * Piskvork engines (via PiskvorkBridgeAI, which spawns each command as a
 * subprocess speaking the Gomocup protocol on stdin/stdout), or between an
 * engine and Ludii's built-in RandomAI.
 *
 * Both sides speak the SAME outer-open rule enforced by Ludii (.lud), so any
 * Piskvork-compatible third-party engine works even if it does not know the
 * opening restriction: illegal first moves are filtered here (we only offer
 * legal moves... engines that answer with an illegal move get a fallback).
 *
 * Usage:
 *   java LudiiMatch GAMES SECONDS
 *   java LudiiMatch GAMES SECONDS LUDPATH ENG_A ENG_B
 * ENG_A/ENG_B are shell-free token lists; use "random" for Ludii's RandomAI.
 */
public class LudiiMatch {

    public static void main(final String[] args) throws Exception {
        final int numGames = args.length > 0 ? Integer.parseInt(args[0]) : 1;
        final double secs = args.length > 1 ? Double.parseDouble(args[1]) : 2.0;

        final String ludPath = args.length > 2 ? args[2]
                : "third_party/ludii/outer_gomoku.lud";
        final String engA = args.length > 3 ? args[3]
                : "python3,tools/pbrain_oogzero.py,exports/weight_iter_0.onnx,100";
        final String engB = args.length > 4 ? args[4] : "random";

        final String lud = new String(Files.readAllBytes(Path.of(ludPath)));
        final Object obj = Compiler.compile(new Description(lud),
                new UserSelections(List.of()), new Report(), false);
        final Game game = (Game) obj;
        System.out.println("Loaded game: " + game.name());

        int winsA = 0, winsB = 0, draws = 0;

        for (int g = 0; g < numGames; g++) {
            final boolean aIsBlack = (g % 2 == 0);

            final AI pa = makeAI(engA, aIsBlack);
            final AI pb = makeAI(engB, !aIsBlack);

            final Context context = new Context(game, new Trial(game));
            game.start(context);

            if (pa instanceof PiskvorkBridgeAI) pa.initAI(game, aIsBlack ? 1 : 2);
            else pa.initAI(game, 1);
            if (pb instanceof PiskvorkBridgeAI) pb.initAI(game, aIsBlack ? 2 : 1);
            else pb.initAI(game, 2);

            System.out.println("\n=== Game " + (g + 1)
                    + " (A=" + pa.friendlyName() + " as " + (aIsBlack ? "Black" : "White")
                    + ", B=" + pb.friendlyName() + ") ===");

            int moveCount = 0;
            while (!context.trial().over() && moveCount < 225) {
                final int mover = context.state().mover();
                final AI ai;
                if (mover == 1) ai = aIsBlack ? pa : pb;
                else            ai = aIsBlack ? pb : pa;
                final Move move = ai.selectAction(game, context, secs, -1, -1);
                game.apply(context, move);
                moveCount++;
            }

            final main.Status status = context.trial().status();
            final int winner = (status == null) ? 0 : status.winner();
            System.out.println("Result: winner = P" + winner
                    + " after " + moveCount + " moves");

            // save game record for later viewing (SGF + plain move list)
            final String sgf = toSGF(context, game);
            final java.nio.file.Path sgfPath = Path.of("third_party/ludii/games/game_" + (g + 1) + ".sgf");
            java.nio.file.Files.createDirectories(sgfPath.getParent());
            java.nio.file.Files.writeString(sgfPath, sgf);
            System.out.println("saved: " + sgfPath);

            if (winner == 1) { if (aIsBlack) winsA++; else winsB++; }
            else if (winner == 2) { if (aIsBlack) winsB++; else winsA++; }
            else draws++;

            pa.closeAI();
            pb.closeAI();
        }

        System.out.println("\n==== Summary ====");
        System.out.println("A wins: " + winsA);
        System.out.println("B wins: " + winsB);
        System.out.println("Draws: " + draws);
    }

    /** Renders the finished trial as a Gomoku SGF (viewers: Sabaki, GoGrinder, gomokuconquest, etc.). */
    private static String toSGF(final Context context, final Game game) {
        final StringBuilder sgf = new StringBuilder();
        sgf.append("(;GM[4]FF[4]CA[UTF-8]AP[LudiiMatch]SZ[15]\n");
        final other.trial.Trial trial = context.trial();
        final int winner = trial.status().winner();
        sgf.append("RE[").append(winner == 0 ? "0" : winner == 1 ? "B" : "W").append("]KM[0]\n");
        // move list: alternate B/W, coordinates a..o for both axes from (x,y)
        final StringBuilder moves = new StringBuilder();
        for (int i = 0; i < trial.numMoves(); i++) {
            final Move m = trial.getMove(i);
            final int site = m.actions().get(0).to();
            final int x = site / 15, y = site % 15;
            final char cx = (char) ('a' + x), cy = (char) ('a' + y);
            moves.append(';').append(m.mover() == 1 ? 'B' : 'W')
                 .append('[').append(cx).append(cy).append(']');
            if ((i + 1) % 10 == 0) moves.append('\n');
        }
        sgf.append(moves).append(")\n");
        return sgf.toString();
    }

    private static AI makeAI(final String spec, final boolean isBlack) {
        if (spec.equals("random")) {
            final AI r = new RandomAI();
            r.setFriendlyName("Random");
            return r;
        }
        final List<String> cmd = new ArrayList<>(List.of(spec.split(",")));
        final AI ai = new PiskvorkBridgeAI(cmd);
        ai.setFriendlyName(cmd.get(cmd.size() - 1) + (isBlack ? "(A)" : "(B)"));
        return ai;
    }
}
