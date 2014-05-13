/*
 * MatlabRenamer.java
 * (C) 2008 Santiago Villalba (sdvillal@gmail.com)
 */

/*
 * License comes here.
 */

package matlabrenamer;

import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

import java.io.File;
import java.io.IOException;
import java.util.StringTokenizer;
import java.util.TreeSet;

/**
 * @author Santi Villalba
 * @version $Id: MatlabRenamerCLI.java 1296 2009-02-12 06:24:24Z santi $
 */
public class MatlabRenamerCLI{

  @Option(name = "-srcDir", usage = "The source dir", required = true)
  private File _srcDir = null;

  @Option(name = "-dstDir", usage = "Target dir")
  private File _targetDir = null;

  @Option(name = "-prefix", usage = "Prefix")
  private String _prefix = null;

  @Option(name = "-suffix", usage = "Suffix")
  private String _suffix = null;

  @Option(name = "-r", usage = "Subdirectory recursion")
  private boolean _recursive = false;

  @Option(name = "-exclude", usage = "Comma separated list of functions to exclude from renaming")
  private String _exclude = null;

  @Option(name = "-scriptsOnly", usage = "Only make functions out of scripts and do nothing else")
  private boolean _scriptsOnly = false;

  /** The extension to the saved renamers. */
  public final static String EXTENSION = ".mren";

  public static MatlabRenamerCLI parseOptions(String[] args){

    MatlabRenamerCLI mrc = new MatlabRenamerCLI();

    CmdLineParser parser = new CmdLineParser(mrc);
    try{
      parser.parseArgument(args);

      if(!mrc.getSrcDir().exists() || !mrc.getSrcDir().isDirectory())
        throw new Exception("The source dir does not exists!");

      if(!mrc.getSrcDir().exists() || !mrc.getSrcDir().isDirectory())
        throw new Exception("The target dir does not exists!");
    }
    catch(Exception ex){
      System.err.println(ex);
      parser.printUsage(System.err);
      System.exit(-1);
    }

    return mrc;
  }

  public File getSrcDir(){
    return _srcDir;
  }

  public File getTargetDir(){
    return _targetDir;
  }

  public String getPrefix(){
    return _prefix;
  }

  public String getSuffix(){
    return _suffix;
  }

  public boolean isRecursive(){
    return _recursive;
  }

  public String getExclude(){
    return _exclude;
  }

  public boolean isScriptsOnly(){
    return _scriptsOnly;
  }

  public static void main(String[] args) throws IOException, ClassNotFoundException{

    MatlabRenamerCLI mrc = parseOptions(args);

    File srcDir = mrc.getSrcDir();
    File dstDir = mrc.getTargetDir() == null ? srcDir : mrc.getTargetDir();
    MatlabRenamer mp;

    TreeSet<String> exclusionsSet = new TreeSet<String>();
    exclusionsSet.add("Contents");
    String exclusions = mrc.getExclude();
    if(null != exclusions){
      StringTokenizer st = new StringTokenizer(exclusions, ",");
      while(st.hasMoreTokens())
        exclusionsSet.add(st.nextToken());
    }

    if(null == mrc.getTargetDir() || srcDir.equals(dstDir))
      mp = new MatlabRenamer(srcDir, mrc.getPrefix(), mrc.getSuffix(), mrc.isRecursive(), exclusionsSet);
    else
      mp = MatlabRenamer.load(new File(mrc.getSrcDir(), mrc.getSrcDir().getName() + EXTENSION));

    mp.scriptsToFunctions(dstDir, mrc.isRecursive());

    if(!mrc.isScriptsOnly()){
      mp.replaceInDirectory(dstDir, mrc.isRecursive());
      mp.renameInDirectory(dstDir, mrc.isRecursive());
    }

    MatlabRenamer.save(mp, new File(srcDir, srcDir.getName() + EXTENSION));
  }
}
