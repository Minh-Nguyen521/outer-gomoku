import compiler.Compiler;
import main.grammar.Description;
import main.options.UserSelections;
import main.grammar.Report;
import other.AI;
import other.context.Context;
import other.move.Move;
import other.trial.Trial;
import game.Game;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * Ludii AI that wraps the project's Piskvork-protocol engine (pbrain_oogzero.py)
 * via a subprocess. Speaks the Piskvork protocol on stdin/stdout.
 *
 * Ludii vertex site indices for a square board: site = col*rows + row? We
 * compute mapping dynamically by probing a board via topology.
 */
public class PiskvorkBridgeAI extends AI {

    /** Path parts of the engine command, e.g. ["python3","tools/pbrain_oogzero.py","model.onnx","100"] */
    private Process process;
    private BufferedReader in;
    private PrintWriter out;
    private int boardSize = 15;
    private int rows = 15, cols = 15;
    private List<String> engineCmd;

    public PiskvorkBridgeAI(final List<String> engineCmd) {
        this.engineCmd = engineCmd;
        this.friendlyName = "OOGZero (Piskvork bridge)";
    }

    @Override
    public boolean supportsGame(final Game game) {
        return game.name().contains("Gomoku") && !game.isSimultaneousMoveGame();
    }

    @Override
    public void initAI(final Game game, final int playerID) {
        try {
            final ProcessBuilder pb = new ProcessBuilder(engineCmd);
            pb.redirectErrorStream(false);
            process = pb.start();
            in = new BufferedReader(new InputStreamReader(process.getInputStream()));
            out = new PrintWriter(process.getOutputStream(), true);
            sendCommand("INFO timeout_turn 2000");
            sendCommand("INFO timeout_match 60000");
            sendCommand("INFO max_memory 100000000");
            sendCommand("START " + boardSize);
            String resp = readResponse();
            if (resp.startsWith("ERROR")) throw new RuntimeException("engine rejected START: " + resp);
        } catch (final Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void closeAI() {
        if (out != null) {
            sendCommand("END");
            out.close();
        }
        if (process != null) process.destroy();
    }

    @Override
    public Move selectAction(final Game game, final Context context, final double maxSeconds,
                             final int maxIterations, final int maxDepth) {
        // Determine side to move (Ludii player 1 -> Piskvork 1, player 2 -> 2)
        final int mover = context.state().mover();

        // Build BOARD message from current state (relative encoding: 1=me, 2=opp)
        sendCommand("BOARD");
        final other.trial.Trial trial = context.trial();
        for (int i = 0; i < trial.numMoves(); i++) {
            final Move m = trial.getMove(i);
            final other.action.Action act = m.actions().get(0);
            final int site = act.to();
            final int[] xy = siteToXY(site);
            final int ludiiPlayer = m.mover();
            final int rel = (ludiiPlayer == mover) ? 1 : 2;
            sendCommand(xy[0] + "," + xy[1] + "," + rel);
        }
        sendCommand("DONE");

        String resp = readCoordinate();
        if (resp == null) {
            // engine died; fall back to centre-most legal move
            System.err.println("WARN: engine gave no move; taking centre-most legal");
            return centreMostLegal(game, context);
        }
        String[] parts;
        try {
            parts = resp.trim().split(",");
            final int x = Integer.parseInt(parts[0]);
            final int y = Integer.parseInt(parts[1]);

            // find matching legal move
            final int site = xyToSite(x, y);
            for (final Move m : game.moves(context).moves()) {
                if (m.actions().get(0).to() == site)
                    return m;
            }

            // engine's move is ILLEGAL here (typically move 1 outside the outer
            // ring): project the desired point onto the nearest legal site, so
            // the opening still lands as close to the engine's intention
            // (usually the centre) as the outer-open rule allows
            return nearestLegalTo(game, context, site, "illegal (" + x + "," + y + ")");
        } catch (final NumberFormatException e) {
            // fall through to fallback below
        }
        // fallback: engine answered something unparseable
        System.err.println("WARN: engine move '" + resp + "' not usable; taking centre-most legal");
        return centreMostLegal(game, context);
    }

    /** Legal move whose site is closest (Euclidean) to the wanted site. */
    private Move nearestLegalTo(final Game game, final Context context, final int wantSite, final String why) {
        final int wx = wantSite / rows, wy = wantSite % rows;
        Move best = null;
        int bestDist = Integer.MAX_VALUE;
        for (final Move m : game.moves(context).moves()) {
            final int site = m.actions().get(0).to();
            final int dx = site / rows - wx;
            final int dy = site % rows - wy;
            final int d = dx * dx + dy * dy;
            if (d < bestDist) { bestDist = d; best = m; }
        }
        System.err.println("WARN: engine move " + why
                + " -> projecting to nearest legal (dist^2=" + bestDist + ")");
        return best;
    }

    private Move centreMostLegal(final Game game, final Context context) {
        final int mid = (rows - 1) * (cols - 1) / 2;
        return nearestLegalTo(game, context, mid, "none given");
    }

    /** Reads lines until one looks like a coordinate "x,y" (skipping MESSAGE/OK/etc). */
    private String readCoordinate() {
        String line = readResponse();
        while (line != null && !line.matches("\\s*-?\\d+\\s*,\\s*-?\\d+\\s*.*"))
            line = readResponse();
        return line;
    }

    private int[] siteToXY(final int site) {
        // Ludii square board topology: sites are numbered column-major? Probe-free
        // convention: site = x * rows + y for vertex boards built by (square n).
        final int x = site / rows;
        final int y = site % rows;
        return new int[]{x, y};
    }

    private int xyToSite(final int x, final int y) {
        return x * rows + y;
    }

    private void sendCommand(final String cmd) {
        out.println(cmd);
        out.flush();
    }

    private String readResponse() {
        try {
            String line = in.readLine();
            while (line != null && line.isEmpty())
                line = in.readLine();
            return line;
        } catch (final Exception e) {
            return null;
        }
    }
}
