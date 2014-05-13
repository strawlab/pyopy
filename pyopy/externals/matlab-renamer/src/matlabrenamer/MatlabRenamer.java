/*
 * MatlabRenamer.java
 * (C) 2008 Santiago Villalba (sdvillal@gmail.com)
 */

/*
 * License comes here.
 */

package matlabrenamer;

import java.io.*;
import java.util.*;
import java.util.regex.Pattern;

/**
 * Ad-hoc class to rename matlab functions.
 *
 * @author Santi Villalba
 * @version $Id: MatlabRenamer.java 1296 2009-02-12 06:24:24Z santi $
 */
public class MatlabRenamer implements Serializable{

  /** The prefix. */
  protected String _prefix = "";

  /** The suffix. */
  protected String _suffix = "";

  /** The map of functions whose name has been changed. */
  protected Map<String, String> _functionMap;

  /** Functions not to be renamed. */
  protected Set<String> _functionExclusions;

  public MatlabRenamer(File dir, String prefix, String suffix, boolean recursive, Set<String> exclusions) throws IOException{
    _prefix = prefix == null ? "" : prefix;
    _suffix = suffix == null ? "" : suffix;
    _functionExclusions = exclusions == null ? new TreeSet<String>() : exclusions;
    _functionMap = analyzePath(dir, recursive, null);
  }

  public Map<String, String> analyzePath(File dir, boolean recursive, Map<String, String> map) throws IOException{

    if(null == map)
      map = new TreeMap<String, String>();

    for(File file : dir.listFiles()){

      if(isNonClassDirectory(file) && recursive)
        analyzePath(file, true, map);

      if(isFunction(file)){

        String name = removeExtension(file.getName());
        if(!_functionExclusions.contains(name)){
          String newName = name;

          if(!name.startsWith(_prefix))
            newName = _prefix + newName;
          if(!name.endsWith(_suffix))
            newName = newName + _suffix;

          map.put(name, newName);
        }
      }
    }

    return map;
  }

  /**
   * Get the extension of the file.
   *
   * @param fileName the file name
   *
   * @return the extension of the file
   */
  public static String getExtension(String fileName){
    int pointIndex = fileName.lastIndexOf(".");
    if(-1 == pointIndex)
      return "";
    return fileName.substring(pointIndex + 1);
  }

  /**
   * Return the name of a file without the extension.
   *
   * @param fileName the file name
   *
   * @return the name of a file without the extension.
   */
  public static String removeExtension(String fileName){
    int pointIndex = fileName.lastIndexOf(".");
    if(-1 == pointIndex)
      return fileName;
    return fileName.substring(0, pointIndex);
  }

  public static boolean isNonClassDirectory(File file){

    return file.isDirectory() && !file.getName().startsWith("@");
  }

  /**
   * Is the file a matlab ".m" file?
   *
   * @param file the file
   *
   * @return <code>true</code> iff the file is an "m" file
   */
  public static boolean isMFile(File file){

    return !file.isDirectory() && getExtension(file.getName()).equals("m");
  }

  /**
   * Is the file a matlab function?
   *
   * @param file the file
   *
   * @return <code>true</code> iff the file contains a valid matlab function
   */
  public static boolean isFunction(File file) throws IOException{

    BufferedReader br = null;

    try{
      if(!isMFile(file))
        return false;

      br = new BufferedReader(new FileReader(file));
      String line = br.readLine();
      while(line != null){
        StringTokenizer st = new StringTokenizer(line);
        String token = null;

        if(st.hasMoreTokens())
          token = st.nextToken();

        if(null == token || token.startsWith("%"))  //It is a comment or a blank line
          line = br.readLine();
        else
          return token.equals("function");
      }

      return false;
    }
    finally{
      if(br != null)
        br.close();
    }
  }

  /**
   * Rename all usages of the functions. It is very naive at the moment and can easily fail.
   * <p/>
   * Right now it is done the easy way, we don't check if this are actual function calls and can be nasty in comments
   * (no way of avoiding it unless asking the user or using ai)...
   * <p/>
   * Note: the function name is always the name of the m file. From the matlab documentation:
   * <p/>
   * <i>If the filename and the function definition line name are different,
   * the internal (function) name is ignored.</i>
   *
   * @param file the file
   *
   * @return the text file as a string with the replacements done
   *
   * @throws FileNotFoundException
   */
  // TODO: very ad-hoc, but who needs more?
  // TODO: what about continuation lines? $ ...
  // TODO: take into account packages & classes
  // TODO: what about mex files?
  // TODO: what if somewhere this is a variable shadowing the name of the function? easy to catch too, but to be completely safe and fast we would need a full blown grammar (see the downloaded pdf, get the bisonone for octave)...
  public String replaceInOneFile(File file) throws IOException{

    StringBuffer sb = new StringBuffer();
    BufferedReader br = new BufferedReader(new FileReader(file));

    //Lets compile the funtion calls recognizing automatons only once
    TreeMap<String, Pattern> patterns = new TreeMap<String, Pattern>();
    for(String function : _functionMap.keySet()){
      String functionCall = "(?<=^|[^_a-zA-Z0-9])((?i)" + function + ")(?=[^_a-zA-Z0-9]|$)";
      patterns.put(function, Pattern.compile(functionCall));
    }

    String line; //TODO: do this better, faster
    while((line = br.readLine()) != null){
      for(String function : patterns.keySet())
        line = patterns.get(function).matcher(line).replaceAll(_functionMap.get(function));
      sb.append(line).append("\n");
    }

    br.close();

    return sb.toString();
  }

  /**
   * Method for converting a script m-file to a function without parameters
   *
   * @param file the file
   *
   * @return the text file as a string starting with the function clause
   *
   * @throws FileNotFoundException
   */

  public String addFunction(File file) throws IOException{

    StringBuffer sb = new StringBuffer();
    BufferedReader br = new BufferedReader(new FileReader(file));

    String line;

    sb.append("function ").append(removeExtension(file.getName())).append("\n");

    while((line = br.readLine()) != null)
      sb.append(line).append("\n");

    br.close();

    return sb.toString();
  }


  /**
   * Rename all the functions contained in the function list within a directory by prefixing them.
   *
   * @param dir       the directory
   * @param recursive recurse over subdirectories
   *
   * @throws IOException
   */
  public void renameInDirectory(File dir, boolean recursive) throws IOException{

    for(File file : dir.listFiles()){

      if(isNonClassDirectory(file) && recursive)
        renameInDirectory(file, true);

      String name = removeExtension(file.getName());
      if(isFunction(file) && _functionMap.keySet().contains(name)){
        if(!file.renameTo(new File(dir, _functionMap.get(name) + ".m")))
          System.err.println("Error renaming file: " + name);
        //Now check for other files with the same name and different extension - TODO: recheck this, what is some c file references other c files? what about recompilation?
        for(File file2 : dir.listFiles()){
          String name2 = removeExtension(file2.getName());
          if(name2.equals(name))
            if(!file2.renameTo(new File(dir, _functionMap.get(name) + "." + getExtension(file2.getName()))))
              System.err.println("Error renaming file: " + name);
        }
      }
    }
  }

  /**
   * Replace all the calls to the functions in the function list with the new prefixed name.
   *
   * @param dir       the directory
   * @param recursive recurse subdirectories?
   *
   * @throws IOException
   */
  public void replaceInDirectory(File dir, boolean recursive) throws IOException{
    for(File file : dir.listFiles()){
      if(file.isDirectory() && recursive)
        replaceInDirectory(file, true);
      if(isMFile(file)){
        String newMFile = replaceInOneFile(file);
        PrintWriter pw = new PrintWriter(new FileWriter(file));
        pw.print(newMFile);
        pw.close();
      }
    }
  }

  /**
   * Make all scripts functions
   *
   * @param dir       the directory
   * @param recursive recurse subdirectories?
   *
   * @throws IOException
   */
  public void scriptsToFunctions(File dir, boolean recursive) throws IOException{
    for(File file : dir.listFiles()){
      if(file.isDirectory() && recursive)
        replaceInDirectory(file, true);
      if(isMFile(file) && !isFunction(file)){
        String newMFile = addFunction(file);
        PrintWriter pw = new PrintWriter(new FileWriter(file));
        pw.print(newMFile);
        pw.close();
      }
    }
  }

  /**
   * Save a prefixier via seraliation.
   *
   * @param prexifier a prefixier
   * @param dest      the destination file
   *
   * @throws IOException
   */
  public static void save(MatlabRenamer prexifier, File dest) throws IOException{
    ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(dest));
    oos.writeObject(prexifier);
    oos.close();
  }

  /**
   * Load a prefixier saved via serialization.
   *
   * @param src the source file
   *
   * @return the prefixier
   *
   * @throws IOException
   * @throws ClassNotFoundException
   */
  public static MatlabRenamer load(File src) throws IOException, ClassNotFoundException{
    ObjectInputStream ois = new ObjectInputStream(new FileInputStream(src));
    MatlabRenamer pr = (MatlabRenamer)ois.readObject();
    ois.close();
    return pr;
  }
}

/*
TODO: do it one by one and throw exception if it is not the same renaming (weak name collision management)
public void merge(MatlabRenamer another){

  TreeMap<String, String> fm = new TreeMap<String, String>(_functionMap);

  fm.putAll(another._functionMap);

  int size1 = _functionMap.size();
  int size2 = another._functionMap.size();

  if(size1 + size2 != fm.size())
    System.err.println("Warning!!! Merged with a map that contained similar keys!");

  _functionMap = fm;
  */